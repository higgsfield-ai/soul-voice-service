"""Stage-3: map a natural-language voice description into the Stage-1 latents.

The projector reads the backbone's *own* instruction representation — the text-encoder
hidden states it already projects into the backbone's embedding space — and
resamples the instruction span into the same ``z_id`` / ``z_style`` shapes the
audio encoder produces. Nothing downstream changes: the output is a
:class:`VoiceCondition`, so identity still enters through the KV prefix and
style through AdaRMSNorm exactly as in Stage 2.

Deterministic by design (spec 1). One description maps to one voice here; the
one-to-many problem belongs to the later stochastic prior, which can sample in
this same latent space without touching the backbone again.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from .condition import VoiceCondition

CHECKPOINT_FORMAT = "breeze-instruction-projector-stage3-v1"

#: Conditioning modes, in the order the trainer encodes them.
MODES = ("clone", "direction", "design", "null")
MODE_INDEX = {name: position for position, name in enumerate(MODES)}


@dataclass
class InstructionProjectorConfig:
    input_dim: int = 2048  # backbone hidden size
    projector_dim: int = 768
    projector_layers: int = 2
    num_heads: int = 8
    num_id_queries: int = 8
    num_style_queries: int = 4
    output_dim: int = 512
    dropout: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


class _ResamplerBlock(nn.Module):
    """Cross-attention from the learned queries into the instruction tokens."""

    def __init__(self, config: InstructionProjectorConfig) -> None:
        super().__init__()
        dim = config.projector_dim
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class InstructionProjector(nn.Module):
    """Instruction hidden states -> ``z_id`` / ``z_style`` (spec 1)."""

    def __init__(self, config: InstructionProjectorConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.projector_dim
        total_queries = config.num_id_queries + config.num_style_queries

        self.input_proj = nn.Linear(config.input_dim, dim)
        self.queries = nn.Parameter(torch.randn(total_queries, dim) * 0.02)
        self.blocks = nn.ModuleList(
            _ResamplerBlock(config) for _ in range(config.projector_layers)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.id_head = nn.Linear(dim, config.output_dim)
        self.style_head = nn.Linear(dim, config.output_dim)

    def attach_voicebook(self, voicebook) -> None:
        """Route the output through a bank of real speaker latents.

        Kept out of ``__init__`` so a checkpoint trained free-form can be
        promoted to prototype selection without rebuilding the projector, and
        so the bank travels as its own file rather than inside every shard.

        Registered as a real submodule so ``.to(device)`` carries the bank with
        it; its buffers are non-persistent, so this costs nothing in the
        checkpoint.
        """
        self.add_module("_voicebook", voicebook)

    @property
    def voicebook(self):
        return self._modules.get("_voicebook")

    def forward(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
        *,
        sample: bool = False,
        top_k: int | None = None,
        temperature_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """``hidden_states`` is [B, T, input_dim]; ``instruction_mask`` is [B, T]."""
        if instruction_mask.dtype != torch.bool:
            instruction_mask = instruction_mask.bool()
        # A row with no instruction span would make every attention logit -inf
        # and produce NaNs, so give it a single (zeroed) slot to attend to.
        empty = ~instruction_mask.any(dim=-1)
        if bool(empty.any()):
            instruction_mask = instruction_mask.clone()
            instruction_mask[empty, 0] = True
            hidden_states = torch.where(
                empty.view(-1, 1, 1),
                torch.zeros_like(hidden_states),
                hidden_states,
            )

        memory = self.input_proj(hidden_states.to(self.input_proj.weight.dtype))
        queries = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, ~instruction_mask)
        queries = self.output_norm(queries)

        id_queries = queries[:, : self.config.num_id_queries]
        style_queries = queries[:, self.config.num_id_queries :]
        id_tokens = self.id_head(id_queries)
        style_tokens = self.style_head(style_queries)
        output = {
            "id_tokens": id_tokens,
            "style_tokens": style_tokens,
            "id_global": id_tokens.mean(dim=1),
            "style_global": style_tokens.mean(dim=1),
        }
        if self.voicebook is None:
            return output

        # With a bank attached the free-form heads become queries: what the
        # caption asks for, rather than the latent it gets. The latent it gets
        # is a real speaker's, which is what keeps it on the manifold the
        # backbone was taught to read.
        output["id_query"] = output["id_global"]
        output["style_query"] = output["style_global"]
        selected_id, output["id_logits"] = self.voicebook.select(
            output["id_query"], kind="id", sample=sample,
            top_k=top_k, temperature_scale=temperature_scale,
        )
        selected_style, output["style_logits"] = self.voicebook.select(
            output["style_query"], kind="style", sample=sample,
            top_k=top_k, temperature_scale=temperature_scale,
        )
        output["id_tokens"] = selected_id.to(id_tokens.dtype)
        output["style_tokens"] = selected_style.to(style_tokens.dtype)
        output["id_global"] = output["id_tokens"].mean(dim=1)
        output["style_global"] = output["style_tokens"].mean(dim=1)
        return output

    # ------------------------------------------------------------ persistence

    def save(self, path: Path, *, metadata: dict | None = None) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value.cpu() for key, value in self.state_dict().items()},
            path / "instruction_projector.pt",
        )
        if self.voicebook is not None:
            self.voicebook.save(path / "voicebook.pt")
        (path / "instruction_projector.json").write_text(
            json.dumps(
                {
                    "format": CHECKPOINT_FORMAT,
                    "config": self.config.to_json(),
                    "metadata": metadata or {},
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "InstructionProjector":
        path = Path(path)
        payload = json.loads((path / "instruction_projector.json").read_text())
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unexpected checkpoint format {payload.get('format')!r}")
        projector = cls(InstructionProjectorConfig(**payload["config"]))
        projector.load_state_dict(
            torch.load(path / "instruction_projector.pt", map_location="cpu"),
            strict=True,
        )
        if (path / "voicebook.pt").exists():
            from .voicebook import PrototypeVoicebook

            projector.attach_voicebook(PrototypeVoicebook.load(path / "voicebook.pt"))
        return projector


def instruction_token_ids(tokenizer) -> tuple[int, int]:
    """The ids of the two delimiters that bracket the instruction."""
    from .serialization import INS_BOS, INS_EOS

    begin = tokenizer.convert_tokens_to_ids(INS_BOS)
    end = tokenizer.convert_tokens_to_ids(INS_EOS)
    if begin is None or end is None or begin < 0 or end < 0:
        raise ValueError("tokenizer does not define the instruction delimiters")
    return int(begin), int(end)


def instruction_span_mask(
    input_ids: torch.Tensor, *, begin_token_id: int, end_token_id: int
) -> torch.Tensor:
    """Mark the tokens strictly between ``<ins_bos>`` and ``<ins_eos>``.

    the backbone serialises the instruction inside the target text segment, so the
    span is recovered from the delimiters rather than tracked separately
    through collation.
    """
    began = (input_ids == begin_token_id).cumsum(dim=-1) > 0
    ended = (input_ids == end_token_id).cumsum(dim=-1) > 0
    return began & ~ended & (input_ids != begin_token_id)


def compose_by_mode(
    audio: dict[str, torch.Tensor],
    text: dict[str, torch.Tensor],
    mode: torch.Tensor,
    *,
    style_mix_alpha: float = 1.0,
) -> "VoiceCondition":
    """Route each row's identity and style to the source its mode asks for.

    ``clone`` keeps both from the recording, ``direction`` keeps the recording's
    identity but moves delivery toward the instruction by ``style_mix_alpha``,
    ``design`` takes both from the instruction, and ``null`` drops conditioning
    entirely. One batch can hold all four.
    """
    dtype = text["style_global"].dtype
    designed = mode == MODE_INDEX["design"]
    directed = mode == MODE_INDEX["direction"]

    id_tokens = torch.where(
        designed.view(-1, 1, 1), text["id_tokens"], audio["id_tokens"]
    )
    row_alpha = torch.zeros(mode.shape[0], dtype=dtype, device=mode.device)
    row_alpha[directed] = style_mix_alpha
    row_alpha[designed] = 1.0
    blended = mix_style(audio, text, row_alpha)

    conditioned = mode != MODE_INDEX["null"]
    return VoiceCondition(
        id_tokens=id_tokens,
        style_tokens=blended["style_tokens"],
        style_global=blended["style_global"],
        valid_id=conditioned,
        valid_style=conditioned.clone(),
    )


def mix_style(
    reference: dict[str, torch.Tensor],
    text: dict[str, torch.Tensor],
    alpha: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Blend reference and instruction style (spec 4).

    ``alpha = 1`` is a full instruction override, ``alpha = 0`` keeps the
    reference's delivery. Exposed as an inference control for weak versus
    strong style direction.
    """
    if isinstance(alpha, torch.Tensor):
        token_alpha = alpha.view(-1, 1, 1)
        global_alpha = alpha.view(-1, 1)
    else:
        token_alpha = global_alpha = alpha
    return {
        "style_tokens": (1 - token_alpha) * reference["style_tokens"]
        + token_alpha * text["style_tokens"],
        "style_global": (1 - global_alpha) * reference["style_global"]
        + global_alpha * text["style_global"],
    }


def _gather_across_ranks(tensor: torch.Tensor) -> torch.Tensor:
    """Collect every rank's copy, keeping gradients on the local shard.

    Contrastive losses are only as good as their negative pool, and a
    per-device microbatch is a small one. The remote shards arrive without
    gradients, which is the standard trade: they widen the pool of negatives
    without needing a backward pass across the process group.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    buffer = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(buffer, tensor.contiguous())
    buffer[dist.get_rank()] = tensor
    return torch.cat(buffer, dim=0)


def contrastive_alignment_loss(
    text: dict[str, torch.Tensor],
    audio: dict[str, torch.Tensor],
    *,
    keep: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Alignment *with negatives*, which is what stops the projector collapsing.

    :func:`alignment_loss` only pulls each caption's latent onto its own
    recording. When many speakers share a caption — and in a movie corpus most
    do — the cosine is minimised by predicting their average, so every
    description lands on one generic voice. Measured on 120 held-out speakers,
    the projector's identity latents spread over an effective rank of 3.9
    against the audio encoder's 11.0.

    Requiring each caption to prefer its own recording *over other recordings
    in the batch* supplies the missing repulsion. The captions stay ambiguous,
    so the targets are noisy by nature; this is a pressure toward spread, not a
    claim that a description names exactly one speaker.
    """
    losses = {}
    for name in ("id", "style"):
        local_text = nn.functional.normalize(text[f"{name}_global"].float(), dim=-1)
        local_audio = nn.functional.normalize(
            audio[f"{name}_global"].detach().float(), dim=-1
        )
        all_audio = _gather_across_ranks(local_audio)
        offset = 0
        if all_audio.shape[0] != local_audio.shape[0]:
            offset = dist.get_rank() * local_text.shape[0]

        logits = local_text @ all_audio.T / temperature
        targets = torch.arange(
            local_text.shape[0], device=logits.device
        ) + offset
        per_sample = nn.functional.cross_entropy(logits, targets, reduction="none")
        if keep is not None:
            weight = keep.to(per_sample.dtype)
            losses[name] = (per_sample * weight).sum() / weight.sum().clamp_min(1.0)
        else:
            losses[name] = per_sample.mean()
    return losses


def alignment_loss(
    text: dict[str, torch.Tensor],
    audio: dict[str, torch.Tensor],
    *,
    keep: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Pull pooled text latents onto their audio counterparts (spec 2).

    Pooled, not token-by-token: the two encoders are free to order and
    specialise their tokens differently as long as the summary lands in the
    same place. The audio side is detached so the alignment moves the
    projector onto the audio latent space rather than dragging the audio
    encoder toward the noisier caption side.
    """
    losses = {}
    for name in ("id", "style"):
        text_pooled = nn.functional.normalize(text[f"{name}_global"], dim=-1)
        audio_pooled = nn.functional.normalize(
            audio[f"{name}_global"].detach(), dim=-1
        )
        per_sample = 1.0 - (text_pooled * audio_pooled).sum(-1)
        if keep is not None:
            weight = keep.to(per_sample.dtype)
            denominator = weight.sum().clamp_min(1.0)
            losses[name] = (per_sample * weight).sum() / denominator
        else:
            losses[name] = per_sample.mean()
    return losses


__all__ = [
    "instruction_token_ids",
    "CHECKPOINT_FORMAT",
    "MODES",
    "MODE_INDEX",
    "InstructionProjector",
    "InstructionProjectorConfig",
    "alignment_loss",
    "compose_by_mode",
    "contrastive_alignment_loss",
    "instruction_span_mask",
    "mix_style",
]

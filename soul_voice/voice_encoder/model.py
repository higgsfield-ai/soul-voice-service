"""Stage-1 multi-token voice representation (``z_id`` / ``z_style``)."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .frontend import ReferenceFrontend
from .style_features import IDENTITY_PROBE_NAMES, STYLE_TARGET_NAMES

CHECKPOINT_FORMAT = "breeze-voice-encoder-stage1-v1"


@dataclass
class VoiceEncoderConfig:
    input_dim: int = 512
    ref_dim: int = 768
    ref_layers: int = 4
    ref_heads: int = 12
    ref_ffn_dim: int = 3072
    dropout: float = 0.05
    cond_dim: int = 512
    num_id_tokens: int = 8
    num_style_tokens: int = 4
    resampler_layers: int = 2
    resampler_heads: int = 8
    max_frames: int = 1024
    feature_source: str = "tokenizer_prequant_latent"

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Stage1HeadConfig:
    id_embed_dim: int = 256
    num_speakers: int = 0
    num_style_classes: int = 0
    num_style_targets: int = len(STYLE_TARGET_NAMES)
    num_probe_targets: int = len(IDENTITY_PROBE_NAMES)
    hidden_dim: int = 512
    dropout: float = 0.05
    am_softmax_margin: float = 0.2
    am_softmax_scale: float = 30.0

    def to_json(self) -> dict:
        return asdict(self)


def _sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding


class PerceiverResampler(nn.Module):
    """Compress a variable-length reference into a fixed bank of latent tokens."""

    def __init__(
        self,
        *,
        num_queries: int,
        cond_dim: int,
        context_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, cond_dim) * 0.02)
        self.cross_attention = nn.ModuleList()
        self.cross_norm_q = nn.ModuleList()
        self.cross_norm_kv = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.self_norm = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_norm_q.append(nn.LayerNorm(cond_dim))
            self.cross_norm_kv.append(nn.LayerNorm(context_dim))
            self.cross_attention.append(
                nn.MultiheadAttention(
                    embed_dim=cond_dim,
                    num_heads=num_heads,
                    kdim=context_dim,
                    vdim=context_dim,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.self_norm.append(nn.LayerNorm(cond_dim))
            self.self_attention.append(
                nn.MultiheadAttention(
                    embed_dim=cond_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ffn_norm.append(nn.LayerNorm(cond_dim))
            self.ffn.append(
                nn.Sequential(
                    nn.Linear(cond_dim, 4 * cond_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * cond_dim, cond_dim),
                )
            )
        self.output_norm = nn.LayerNorm(cond_dim)
        self.output_proj = nn.Linear(cond_dim, cond_dim)

    def forward(self, context: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        latents = self.queries.unsqueeze(0).expand(batch, -1, -1)
        for index in range(len(self.cross_attention)):
            normalized = self.cross_norm_q[index](latents)
            keys = self.cross_norm_kv[index](context)
            attended, _ = self.cross_attention[index](
                normalized, keys, keys, key_padding_mask=key_padding_mask, need_weights=False
            )
            latents = latents + attended
            normalized = self.self_norm[index](latents)
            attended, _ = self.self_attention[index](
                normalized, normalized, normalized, need_weights=False
            )
            latents = latents + attended
            latents = latents + self.ffn[index](self.ffn_norm[index](latents))
        return self.output_proj(self.output_norm(latents))


class VoiceEncoder(nn.Module):
    """Reference-voice encoder producing ``z_id`` and ``z_style`` token banks.

    The frozen tokenizer frontend is optional so the trainer can share one
    frontend across ranks; when absent, ``forward`` expects pre-extracted
    tokenizer features.
    """

    def __init__(
        self,
        config: VoiceEncoderConfig | None = None,
        *,
        frontend: ReferenceFrontend | None = None,
    ) -> None:
        super().__init__()
        self.config = config or VoiceEncoderConfig()
        # Held as a plain attribute, not a submodule: the tokenizer is a frozen
        # external dependency, so it must stay out of state_dict, parameters()
        # and device moves. Registering it would put ~130 tokenizer tensors in
        # every Stage-1 checkpoint and break strict loading.
        object.__setattr__(self, "frontend", frontend)
        cfg = self.config

        self.input_norm = nn.LayerNorm(cfg.input_dim)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.ref_dim)
        self.position_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "positions",
            _sinusoidal_positions(cfg.max_frames, cfg.ref_dim, torch.device("cpu")),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.ref_dim,
            nhead=cfg.ref_heads,
            dim_feedforward=cfg.ref_ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.reference_encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.ref_layers, norm=nn.LayerNorm(cfg.ref_dim)
        )
        shared = {
            "cond_dim": cfg.cond_dim,
            "context_dim": cfg.ref_dim,
            "num_layers": cfg.resampler_layers,
            "num_heads": cfg.resampler_heads,
            "dropout": cfg.dropout,
        }
        self.id_resampler = PerceiverResampler(num_queries=cfg.num_id_tokens, **shared)
        self.style_resampler = PerceiverResampler(num_queries=cfg.num_style_tokens, **shared)

    def encode_reference(
        self, features: torch.Tensor, frame_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = features.shape[1]
        if frames > self.positions.shape[0]:
            self.positions = _sinusoidal_positions(
                frames, self.config.ref_dim, features.device
            )
        hidden = self.input_proj(self.input_norm(features))
        hidden = hidden + self.position_scale * self.positions[:frames].to(hidden.dtype)
        padding_mask = torch.arange(frames, device=features.device).unsqueeze(
            0
        ) >= frame_lengths.unsqueeze(1)
        hidden = self.reference_encoder(hidden, src_key_padding_mask=padding_mask)
        return hidden, padding_mask

    def forward(
        self,
        reference_audio: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        *,
        features: torch.Tensor | None = None,
        frame_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``id_tokens``, ``style_tokens`` and ``style_global``.

        Pass ``reference_audio`` (``[B, N]`` at 24 kHz) for the standalone path,
        or pre-extracted tokenizer ``features`` to reuse a shared frontend.
        """
        if features is None:
            if reference_audio is None:
                raise ValueError("provide either reference_audio or features")
            if self.frontend is None:
                raise ValueError("no frozen frontend attached; pass features instead")
            if lengths is None:
                lengths = torch.full(
                    (reference_audio.shape[0],),
                    reference_audio.shape[1],
                    device=reference_audio.device,
                    dtype=torch.long,
                )
            features, frame_lengths = self.frontend(reference_audio, lengths)
        if frame_lengths is None:
            frame_lengths = torch.full(
                (features.shape[0],), features.shape[1], device=features.device, dtype=torch.long
            )
        features = features.to(self.input_proj.weight.dtype)
        hidden, padding_mask = self.encode_reference(features, frame_lengths)
        id_tokens = self.id_resampler(hidden, padding_mask)
        style_tokens = self.style_resampler(hidden, padding_mask)
        return {
            "id_tokens": id_tokens,
            "style_tokens": style_tokens,
            "id_global": id_tokens.mean(dim=1),
            "style_global": style_tokens.mean(dim=1),
        }

    def save_stage1(self, path: Path, *, metadata: dict | None = None) -> Path:
        """Persist the Stage-1 encoder independently of the backbone."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {key: value.cpu() for key, value in self.state_dict().items()}
        torch.save(state, path / "voice_encoder.pt")
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": self.config.to_json(),
            "metadata": metadata or {},
        }
        (path / "voice_encoder.json").write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def load_stage1(
        cls, path: Path, *, frontend: ReferenceFrontend | None = None
    ) -> "VoiceEncoder":
        path = Path(path)
        payload = json.loads((path / "voice_encoder.json").read_text())
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unexpected checkpoint format {payload.get('format')!r}")
        config = VoiceEncoderConfig(**payload["config"])
        model = cls(config)
        state = torch.load(path / "voice_encoder.pt", map_location="cpu")
        model.load_state_dict(state, strict=True)
        if frontend is not None:
            object.__setattr__(model, "frontend", frontend)
        return model


class AMSoftmax(nn.Module):
    """Additive-margin softmax over training speakers (training only)."""

    def __init__(self, embed_dim: int, num_classes: int, *, margin: float, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = nn.functional.normalize(self.weight, dim=-1)
        cosine = nn.functional.linear(nn.functional.normalize(embeddings, dim=-1), weight)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        logits = self.scale * (cosine - self.margin * one_hot)
        return nn.functional.cross_entropy(logits, labels)


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class Stage1Heads(nn.Module):
    """Training-only projection, speaker-metric, style and diagnostic heads."""

    def __init__(self, config: Stage1HeadConfig, *, cond_dim: int = 512) -> None:
        super().__init__()
        self.config = config
        # The speaker metric is applied to pooled z_id directly rather than
        # through a projection head: z_id is the deliverable, and a projection
        # lets the encoder satisfy same-speaker consistency by collapsing z_id
        # while keeping speaker structure only in the discarded head.
        self.adversary_projection = (
            _mlp(cond_dim, config.hidden_dim, config.id_embed_dim, config.dropout)
            if config.num_speakers > 0
            else None
        )
        self.am_softmax = (
            AMSoftmax(
                config.id_embed_dim,
                config.num_speakers,
                margin=config.am_softmax_margin,
                scale=config.am_softmax_scale,
            )
            if config.num_speakers > 0
            else None
        )
        self.style_regression = _mlp(
            cond_dim, config.hidden_dim, config.num_style_targets, config.dropout
        )
        self.style_classifier = (
            _mlp(cond_dim, config.hidden_dim, config.num_style_classes, config.dropout)
            if config.num_style_classes > 0
            else None
        )
        # Diagnostic probes read detached inputs; they never shape the encoder.
        self.probe_style_from_id = _mlp(
            cond_dim, config.hidden_dim, config.num_style_targets, config.dropout
        )
        self.probe_identity_from_id = _mlp(
            cond_dim, config.hidden_dim, config.num_probe_targets, config.dropout
        )
        self.probe_identity_from_style = _mlp(
            cond_dim, config.hidden_dim, config.num_probe_targets, config.dropout
        )

    @staticmethod
    def identity_embedding(id_global: torch.Tensor) -> torch.Tensor:
        """Scoring space for the speaker metric: pooled z_id on the unit sphere."""
        return nn.functional.normalize(id_global, dim=-1)

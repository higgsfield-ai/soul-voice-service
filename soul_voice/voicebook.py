"""A bank of real speaker latents that the instruction projector selects from.

The deterministic projector failed for a structural reason: one caption
honestly describes many voices, so a map trained under cross-entropy either
predicts their average — identity latents collapsing to an effective rank of
3.8 against the audio encoder's 11.0 — or, when pushed apart by contrastive
negatives, spreads into directions the backbone cannot read, leaving 36% of its
energy outside the audio subspace and generated voices no more distinct.

Selecting from real latents removes both failure modes by construction. Every
output is a latent the audio encoder actually produced, so it is on-manifold by
definition, and sampling from the caption-conditional distribution gives one
description many voices instead of their average. Identity and style are drawn
separately, which also lets a caption combine one speaker's timbre with
another's delivery.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VoicebookConfig:
    temperature: float = 0.07
    top_k: int = 64
    gumbel_tau: float = 1.0


class PrototypeVoicebook(nn.Module):
    """Score a query against real speaker latents and return one of them.

    The prototypes are buffers rather than parameters. They are measurements of
    the audio encoder's output, not something to be learned here — letting
    gradients move them would reintroduce the drift off the manifold that this
    module exists to prevent.
    """

    def __init__(
        self,
        id_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        config: VoicebookConfig | None = None,
        id_quality: torch.Tensor | None = None,
        style_quality: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or VoicebookConfig()
        # persistent=False keeps a 200 MB bank out of every checkpoint shard;
        # it is written once alongside the projector instead.
        self.register_buffer("id_tokens", id_tokens, persistent=False)
        self.register_buffer("style_tokens", style_tokens, persistent=False)
        self.register_buffer(
            "id_keys",
            nn.functional.normalize(id_tokens.mean(dim=1), dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "style_keys",
            nn.functional.normalize(style_tokens.mean(dim=1), dim=-1),
            persistent=False,
        )
        # A standing opinion about each prototype, in the same units as the
        # selection logits. The bank was built from whatever speakers the
        # corpus contained, so some prototypes are band-limited, noisy or
        # otherwise poor recordings, and that is a property of the prototype
        # rather than of any caption - measurable once and reusable forever.
        # Zeros leave selection exactly as it was.
        self.register_buffer(
            "id_quality",
            torch.zeros(id_tokens.shape[0]) if id_quality is None else id_quality,
            persistent=False,
        )
        self.register_buffer(
            "style_quality",
            torch.zeros(style_tokens.shape[0])
            if style_quality is None else style_quality,
            persistent=False,
        )

    @property
    def size(self) -> int:
        return int(self.id_tokens.shape[0])

    def logits(self, query: torch.Tensor, *, kind: str) -> torch.Tensor:
        keys = self.id_keys if kind == "id" else self.style_keys
        query = nn.functional.normalize(query.float(), dim=-1)
        return query @ keys.to(query.dtype).T / self.config.temperature

    def select(
        self,
        query: torch.Tensor,
        *,
        kind: str,
        sample: bool,
        top_k: int | None = None,
        temperature_scale: float = 1.0,
        quality_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tokens, logits)`` for one query batch.

        With ``sample`` the pick is a hard draw made differentiable by the
        straight-through Gumbel estimator, so the forward pass carries exactly
        one real speaker's latent while the projector still receives gradient.
        Averaging the top matches instead would put us back where we started.
        """
        bank = self.id_tokens if kind == "id" else self.style_tokens
        logits = self.logits(query, kind=kind)
        limit = self.config.top_k if top_k is None else top_k
        scaled = logits / max(temperature_scale, 1e-6)
        # Applied before the top-k cut, so a prototype known to sound poor is
        # removed from the pool rather than merely made less likely within it.
        # The returned logits stay clean: they are the caption's own opinion and
        # the training loss is entitled to see it unedited.
        bias = self.id_quality if kind == "id" else self.style_quality
        if bias is not None and float(bias.abs().max()) > 0:
            scaled = scaled + quality_weight * bias.to(scaled.dtype)
        if limit and limit < scaled.shape[-1]:
            floor = scaled.topk(limit, dim=-1).values[..., -1:]
            scaled = scaled.masked_fill(scaled < floor, float("-inf"))

        if sample:
            weights = nn.functional.gumbel_softmax(
                scaled, tau=self.config.gumbel_tau, hard=True, dim=-1
            )
        else:
            index = scaled.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(scaled).scatter_(-1, index, 1.0)
            # Straight-through so the deterministic path trains too.
            weights = weights + scaled.softmax(dim=-1) - scaled.softmax(dim=-1).detach()

        flat = bank.reshape(bank.shape[0], -1).to(weights.dtype)
        tokens = (weights @ flat).view(-1, bank.shape[1], bank.shape[2])
        return tokens, logits

    def target_distribution(
        self, latent: torch.Tensor, *, kind: str, temperature: float = 0.05
    ) -> torch.Tensor:
        """Soft labels over prototypes for a known recording.

        Used to teach the caption where in the bank its speaker lives. Soft
        rather than one-hot because neighbouring prototypes are genuinely
        similar voices, and pretending otherwise would make the target noise.
        """
        keys = self.id_keys if kind == "id" else self.style_keys
        latent = nn.functional.normalize(latent.detach().float(), dim=-1)
        return (latent @ keys.to(latent.dtype).T / temperature).softmax(dim=-1)

    # ------------------------------------------------------------ persistence

    def save(self, path) -> None:
        from pathlib import Path

        torch.save(
            {
                "id_tokens": self.id_tokens.cpu(),
                "style_tokens": self.style_tokens.cpu(),
                "config": self.config.__dict__,
                "id_quality": self.id_quality.cpu(),
                "style_quality": self.style_quality.cpu(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path, *, device: str | torch.device = "cpu") -> "PrototypeVoicebook":
        payload = torch.load(path, map_location="cpu")
        book = cls(
            payload["id_tokens"],
            payload["style_tokens"],
            VoicebookConfig(**payload.get("config", {})),
            # Absent in banks written before the audit existed, which is the
            # same as having no opinion about any prototype.
            payload.get("id_quality"),
            payload.get("style_quality"),
        )
        return book.to(device)


def voicebook_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, keep: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy of the caption's prototype distribution against the truth."""
    per_sample = -(targets * logits.log_softmax(dim=-1)).sum(dim=-1)
    if keep is None:
        return per_sample.mean()
    weight = keep.to(per_sample.dtype)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)


__all__ = [
    "PrototypeVoicebook",
    "VoicebookConfig",
    "voicebook_loss",
]

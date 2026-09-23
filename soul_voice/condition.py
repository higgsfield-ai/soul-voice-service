"""The conditioning payload the backbone accepts in Stage 2."""

from dataclasses import dataclass, replace

import torch


@dataclass
class VoiceCondition:
    """Stage-1 latents plus per-sample validity flags.

    ``valid_id`` / ``valid_style`` are what conditioning dropout and the
    unconditional CFG branch toggle. A sample whose flag is ``False`` must come
    out of the backbone exactly as the unmodified pretrained model would, so the flags
    are honoured at the point of use (attention logits, modulation output)
    rather than by zeroing the latents.
    """

    id_tokens: torch.Tensor  # [B, num_id_tokens, cond_dim]
    style_tokens: torch.Tensor  # [B, num_style_tokens, cond_dim]
    style_global: torch.Tensor  # [B, cond_dim]
    valid_id: torch.Tensor  # [B] bool
    valid_style: torch.Tensor  # [B] bool

    def __post_init__(self) -> None:
        batch = self.id_tokens.shape[0]
        for name in ("style_tokens", "style_global", "valid_id", "valid_style"):
            tensor = getattr(self, name)
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"VoiceCondition.{name} has batch {tensor.shape[0]}, expected {batch}"
                )
        if self.id_tokens.ndim != 3 or self.style_tokens.ndim != 3:
            raise ValueError("id_tokens and style_tokens must be [B, N, cond_dim]")
        if self.style_global.ndim != 2:
            raise ValueError("style_global must be [B, cond_dim]")

    @property
    def batch_size(self) -> int:
        return int(self.id_tokens.shape[0])

    @property
    def device(self) -> torch.device:
        return self.id_tokens.device

    def to(self, *, device=None, dtype=None) -> "VoiceCondition":
        def move(tensor: torch.Tensor, cast: bool) -> torch.Tensor:
            return tensor.to(device=device, dtype=dtype if cast else None)

        return VoiceCondition(
            id_tokens=move(self.id_tokens, True),
            style_tokens=move(self.style_tokens, True),
            style_global=move(self.style_global, True),
            valid_id=move(self.valid_id, False),
            valid_style=move(self.valid_style, False),
        )

    def index_select(self, index: torch.Tensor) -> "VoiceCondition":
        """Expand to a different batch axis, e.g. one row per depth-decoder frame."""
        return VoiceCondition(
            id_tokens=self.id_tokens[index],
            style_tokens=self.style_tokens[index],
            style_global=self.style_global[index],
            valid_id=self.valid_id[index],
            valid_style=self.valid_style[index],
        )

    def repeat(self, times: int) -> "VoiceCondition":
        """Tile along the batch axis, for CFG branches stacked into one forward."""
        return VoiceCondition(
            id_tokens=self.id_tokens.repeat(times, 1, 1),
            style_tokens=self.style_tokens.repeat(times, 1, 1),
            style_global=self.style_global.repeat(times, 1),
            valid_id=self.valid_id.repeat(times),
            valid_style=self.valid_style.repeat(times),
        )

    def detach(self) -> "VoiceCondition":
        return VoiceCondition(
            id_tokens=self.id_tokens.detach(),
            style_tokens=self.style_tokens.detach(),
            style_global=self.style_global.detach(),
            valid_id=self.valid_id,
            valid_style=self.valid_style,
        )

    def nulled(self) -> "VoiceCondition":
        """The unconditional CFG branch: same latents, both flags off."""
        return replace(
            self,
            valid_id=torch.zeros_like(self.valid_id),
            valid_style=torch.zeros_like(self.valid_style),
        )

    def with_validity(
        self, *, valid_id: torch.Tensor | None = None, valid_style: torch.Tensor | None = None
    ) -> "VoiceCondition":
        return replace(
            self,
            valid_id=self.valid_id if valid_id is None else valid_id,
            valid_style=self.valid_style if valid_style is None else valid_style,
        )

    @classmethod
    def from_encoder(
        cls, output: dict[str, torch.Tensor], *, valid: torch.Tensor | None = None
    ) -> "VoiceCondition":
        """Wrap a :meth:`VoiceEncoder.forward` result."""
        id_tokens = output["id_tokens"]
        if valid is None:
            valid = torch.ones(
                id_tokens.shape[0], dtype=torch.bool, device=id_tokens.device
            )
        return cls(
            id_tokens=id_tokens,
            style_tokens=output["style_tokens"],
            style_global=output["style_global"],
            valid_id=valid,
            valid_style=valid.clone(),
        )

    @classmethod
    def null(
        cls,
        batch_size: int,
        *,
        cond_dim: int,
        num_id_tokens: int,
        num_style_tokens: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "VoiceCondition":
        """A condition that leaves the backbone untouched."""
        zeros = torch.zeros
        return cls(
            id_tokens=zeros(batch_size, num_id_tokens, cond_dim, device=device, dtype=dtype),
            style_tokens=zeros(
                batch_size, num_style_tokens, cond_dim, device=device, dtype=dtype
            ),
            style_global=zeros(batch_size, cond_dim, device=device, dtype=dtype),
            valid_id=zeros(batch_size, dtype=torch.bool, device=device),
            valid_style=zeros(batch_size, dtype=torch.bool, device=device),
        )


def apply_conditioning_dropout(
    condition: VoiceCondition,
    *,
    drop_id_probability: float,
    drop_style_probability: float,
    drop_both_probability: float,
    generator: torch.Generator | None = None,
) -> "VoiceCondition":
    """Independent identity/style dropout plus a joint-drop branch (spec 11).

    The joint draw is applied on top of the independent ones so the marginal
    drop rate of each stream is ``p_stream + p_both - p_stream * p_both``.
    """
    batch = condition.batch_size
    device = condition.device

    def draw(probability: float) -> torch.Tensor:
        if probability <= 0.0:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        noise = torch.rand(batch, device=device, generator=generator)
        return noise < probability

    drop_both = draw(drop_both_probability)
    drop_id = draw(drop_id_probability) | drop_both
    drop_style = draw(drop_style_probability) | drop_both
    return condition.with_validity(
        valid_id=condition.valid_id & ~drop_id,
        valid_style=condition.valid_style & ~drop_style,
    )


__all__ = ["VoiceCondition", "apply_conditioning_dropout"]

"""New Stage-2 parameters: identity KV projections and the AdaRMSNorm style path."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .attention import MASKED_PREFIX_LOGIT, AttentionPrefixSlot
from .condition import VoiceCondition

LAYER_SELECTIONS = ("all", "every_second_layer", "none")


@dataclass
class VoiceConditionConfig:
    """Every knob spec section 17 leaves to the implementation."""

    cond_dim: int = 512
    num_id_tokens: int = 8
    num_style_tokens: int = 4
    # Length of the KV prefix. ``z_id``'s token count is fixed by the Stage-1
    # encoder, so sweeping the prefix length (spec 14) resamples along the token
    # axis instead of pretending the encoder can be reshaped after the fact.
    prefix_tokens: int | None = None

    identity_prefix: bool = True
    style_adarms: bool = True

    prefix_layers: str = "all"
    prefix_depth_layers: str = "all"
    adarms_layers: str = "all"
    adarms_depth_layers: str = "all"

    prefix_projection_shared_across_layers: bool = False
    prefix_scale_type: str = "scalar_per_layer"  # or scalar_per_head
    prefix_key_norm: bool = True
    gate_init: float = 0.0

    style_pool: str = "mean"  # or learned_attention
    adarms_trunk_dim: int = 1024
    adarms_head: str = "low_rank"  # or full
    adarms_rank: int = 256

    null_condition: str = "zero"  # or learned

    def to_json(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for name, value in (
            ("prefix_layers", self.prefix_layers),
            ("prefix_depth_layers", self.prefix_depth_layers),
            ("adarms_layers", self.adarms_layers),
            ("adarms_depth_layers", self.adarms_depth_layers),
        ):
            if value not in LAYER_SELECTIONS:
                raise ValueError(f"{name}={value!r} must be one of {LAYER_SELECTIONS}")
        if self.prefix_scale_type not in ("scalar_per_layer", "scalar_per_head"):
            raise ValueError(f"unknown prefix_scale_type {self.prefix_scale_type!r}")
        if self.style_pool not in ("mean", "learned_attention"):
            raise ValueError(f"unknown style_pool {self.style_pool!r}")
        if self.adarms_head not in ("full", "low_rank"):
            raise ValueError(f"unknown adarms_head {self.adarms_head!r}")
        if self.null_condition not in ("zero", "learned"):
            raise ValueError(f"unknown null_condition {self.null_condition!r}")
        if not self.identity_prefix and not self.style_adarms:
            raise ValueError("at least one conditioning path must be enabled")


@dataclass(frozen=True)
class TransformerSpec:
    """Shapes of one the backbone transformer stack."""

    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


def select_layers(selection: str, num_layers: int) -> list[int]:
    if selection == "all":
        return list(range(num_layers))
    if selection == "every_second_layer":
        return list(range(0, num_layers, 2))
    if selection == "none":
        return []
    raise ValueError(f"unknown layer selection {selection!r}")


class StyleModulationSlot:
    """Per-layer handoff for the two AdaRMSNorm sites inside a decoder layer."""

    __slots__ = ("attn", "mlp")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.attn = None
        self.mlp = None

    @property
    def active(self) -> bool:
        return self.attn is not None


class AdaRMSNorm(nn.Module):
    """RMSNorm(x) * (1 + gamma(s)) + beta(s), sharing the pretrained weight.

    The pretrained ``weight`` Parameter is adopted by reference under the same
    attribute name, so swapping this module in leaves the backbone ``state_dict``
    keys unchanged and the base checkpoint still saves and loads.
    """

    def __init__(self, base: nn.Module, slot: StyleModulationSlot, kind: str) -> None:
        super().__init__()
        if kind not in ("attn", "mlp"):
            raise ValueError(f"unknown AdaRMSNorm kind {kind!r}")
        self.weight = base.weight
        self.variance_epsilon = float(base.variance_epsilon)
        self.kind = kind
        object.__setattr__(self, "slot", slot)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden = hidden_states.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.variance_epsilon)
        normed = self.weight * hidden.to(input_dtype)

        modulation = getattr(self.slot, self.kind)
        if modulation is None:
            return normed
        gamma, beta = modulation
        gamma = gamma.unsqueeze(-2).to(input_dtype)
        beta = beta.unsqueeze(-2).to(input_dtype)
        return normed * (1.0 + gamma) + beta

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, kind={self.kind}"


class IdentityPrefix(nn.Module):
    """Per-layer ``W_K``/``W_V`` projections of ``z_id`` plus the zero-start gate."""

    def __init__(self, spec: TransformerSpec, config: VoiceConditionConfig, layers: list[int]):
        super().__init__()
        self.spec = spec
        self.config = config
        self.layers = list(layers)
        self.prefix_tokens = config.prefix_tokens or config.num_id_tokens
        num_slots = 1 if config.prefix_projection_shared_across_layers else len(self.layers)

        if self.prefix_tokens == config.num_id_tokens:
            self.resample = None
        else:
            self.resample = nn.Linear(
                config.num_id_tokens, self.prefix_tokens, bias=False
            )
            with torch.no_grad():
                # Start as a plain linear resampling of the identity tokens.
                self.resample.weight.copy_(
                    torch.nn.functional.interpolate(
                        torch.eye(config.num_id_tokens).unsqueeze(0),
                        size=self.prefix_tokens,
                        mode="linear",
                        align_corners=True,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )

        def projection() -> nn.Linear:
            layer = nn.Linear(config.cond_dim, spec.kv_width, bias=False)
            nn.init.normal_(layer.weight, std=0.02)
            return layer

        self.k_proj = nn.ModuleList(projection() for _ in range(num_slots))
        self.v_proj = nn.ModuleList(projection() for _ in range(num_slots))
        if config.prefix_key_norm:
            # Keeps prefix keys on the same scale as real keys, which Qwen3
            # enforces for its own keys with a per-head-dim RMSNorm.
            self.key_norm = nn.ModuleList(
                nn.RMSNorm(spec.head_dim) for _ in range(num_slots)
            )
        else:
            self.key_norm = None

        gate_width = (
            spec.num_attention_heads
            if config.prefix_scale_type == "scalar_per_head"
            else 1
        )
        self.gate = nn.Parameter(
            torch.full((len(self.layers), gate_width), float(config.gate_init))
        )
        if config.null_condition == "learned":
            self.null_tokens = nn.Parameter(
                torch.zeros(self.prefix_tokens, config.cond_dim)
            )
            nn.init.normal_(self.null_tokens, std=0.02)
        else:
            self.null_tokens = None

    def _slot_index(self, position: int) -> int:
        return 0 if self.config.prefix_projection_shared_across_layers else position

    def build(
        self,
        condition: VoiceCondition,
        slots: dict[int, AttentionPrefixSlot],
        *,
        expand_index: torch.Tensor | None = None,
    ) -> None:
        """Project ``z_id`` into every selected layer's KV slot.

        ``expand_index`` maps the condition's batch axis onto the axis the stack
        actually runs on; projecting first and expanding after keeps the matmuls
        on the small axis.
        """
        tokens = condition.id_tokens
        valid = condition.valid_id
        batch = tokens.shape[0]
        if self.resample is not None:
            tokens = self.resample(tokens.transpose(1, 2)).transpose(1, 2)

        bias = None
        if self.null_tokens is None:
            # Masking the logits (rather than the latents) keeps a dropped
            # sample bit-identical to unconditioned the backbone.
            if not bool(valid.all()):
                bias = torch.zeros(
                    batch, 1, 1, tokens.shape[1], device=tokens.device
                )
                bias.masked_fill_(~valid.view(batch, 1, 1, 1), MASKED_PREFIX_LOGIT)
                if expand_index is not None:
                    bias = bias[expand_index]
        else:
            tokens = torch.where(
                valid.view(batch, 1, 1), tokens, self.null_tokens.to(tokens.dtype)
            )

        heads = self.spec.num_key_value_heads
        head_dim = self.spec.head_dim
        for position, layer_idx in enumerate(self.layers):
            slot = slots.get(layer_idx)
            if slot is None:
                continue
            index = self._slot_index(position)
            key = self.k_proj[index](tokens).view(batch, -1, heads, head_dim)
            if self.key_norm is not None:
                key = self.key_norm[index](key)
            value = self.v_proj[index](tokens).view(batch, -1, heads, head_dim)
            if expand_index is not None:
                key = key[expand_index]
                value = value[expand_index]
            gate = self.gate[position].view(1, -1, 1, 1)
            slot.set(
                key.transpose(1, 2),
                value.transpose(1, 2),
                gate,
                None if bias is None else bias.to(key.dtype),
            )


class StyleModulation(nn.Module):
    """Shared style trunk (spec 7) and the zero-initialised per-layer heads (spec 8)."""

    def __init__(self, spec: TransformerSpec, config: VoiceConditionConfig, layers: list[int]):
        super().__init__()
        self.spec = spec
        self.config = config
        self.layers = list(layers)
        width = 4 * spec.hidden_size

        if config.adarms_head == "full":
            self.head = nn.ModuleList(
                nn.Linear(config.adarms_trunk_dim, width) for _ in self.layers
            )
            for head in self.head:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.head_down = None
        else:
            self.head_down = nn.ModuleList(
                nn.Linear(config.adarms_trunk_dim, config.adarms_rank, bias=False)
                for _ in self.layers
            )
            for down in self.head_down:
                nn.init.normal_(down.weight, std=0.02)
            self.head = nn.ModuleList(
                nn.Linear(config.adarms_rank, width) for _ in self.layers
            )
            for head in self.head:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def build(
        self,
        trunk_state: torch.Tensor,
        valid: torch.Tensor,
        slots: dict[int, StyleModulationSlot],
    ) -> None:
        hidden = self.spec.hidden_size
        keep = valid.view(-1, 1).to(trunk_state.dtype)
        for position, layer_idx in enumerate(self.layers):
            slot = slots.get(layer_idx)
            if slot is None:
                continue
            state = trunk_state
            if self.head_down is not None:
                state = self.head_down[position](state)
            modulation = self.head[position](state)
            if self.config.null_condition == "zero":
                # Exact identity for dropped samples, at every training step.
                modulation = modulation * keep
            gamma_attn, beta_attn, gamma_mlp, beta_mlp = modulation.split(hidden, dim=-1)
            slot.attn = (gamma_attn, beta_attn)
            slot.mlp = (gamma_mlp, beta_mlp)


class StyleTrunk(nn.Module):
    """Pools ``z_style`` and maps it to the shared modulation state."""

    def __init__(self, config: VoiceConditionConfig):
        super().__init__()
        self.config = config
        if config.style_pool == "learned_attention":
            self.pool_query = nn.Parameter(torch.zeros(config.cond_dim))
            nn.init.normal_(self.pool_query, std=0.02)
            self.pool_key = nn.Linear(config.cond_dim, config.cond_dim, bias=False)
        else:
            self.pool_query = None
            self.pool_key = None
        self.mlp = nn.Sequential(
            nn.Linear(config.cond_dim, config.adarms_trunk_dim),
            nn.SiLU(),
            nn.Linear(config.adarms_trunk_dim, config.adarms_trunk_dim),
        )
        if config.null_condition == "learned":
            self.null_style = nn.Parameter(torch.zeros(config.cond_dim))
            nn.init.normal_(self.null_style, std=0.02)
        else:
            self.null_style = None

    def pool(self, condition: VoiceCondition) -> torch.Tensor:
        if self.pool_query is None:
            return condition.style_tokens.mean(dim=1)
        keys = self.pool_key(condition.style_tokens)
        scores = torch.einsum("bnd,d->bn", keys, self.pool_query.to(keys.dtype))
        weights = torch.softmax(scores / (keys.shape[-1] ** 0.5), dim=-1)
        return torch.einsum("bn,bnd->bd", weights, condition.style_tokens)

    def forward(self, condition: VoiceCondition) -> torch.Tensor:
        pooled = self.pool(condition)
        if self.null_style is not None:
            pooled = torch.where(
                condition.valid_style.view(-1, 1), pooled, self.null_style.to(pooled.dtype)
            )
        return self.mlp(pooled)


__all__ = [
    "LAYER_SELECTIONS",
    "AdaRMSNorm",
    "IdentityPrefix",
    "StyleModulation",
    "StyleModulationSlot",
    "StyleTrunk",
    "TransformerSpec",
    "VoiceConditionConfig",
    "select_layers",
]

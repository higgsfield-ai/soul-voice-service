"""Voice conditioning wired into the backbone, with the control flow made explicit.

The arithmetic is unchanged and imported as-is; what is rewritten is *when* it
gets applied. The training version worked that out by watching the model at
runtime - which guidance branch it was in, from the identity of the first KV
cache it saw, and which depth rows were still alive, by comparing hidden-state
tensors. Both are silent when wrong: the output is still audio, in the wrong
voice. Here the caller says what it already knows.

    conditioner = VoiceConditioner.load(bundle).attach(model)
    conditioner.set_condition(condition)          # projects once per utterance
    with conditioner.branch("conditional"):
        ...                                       # backbone forward
    with conditioner.branch("unconditional"):
        ...                                       # negative forward, no voice
    conditioner.set_depth_rows(active)            # when rows drop out
"""

import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .attention import (
    ATTENTION_IMPLEMENTATION,
    AttentionPrefixSlot,
    register_attention_implementation,
)
from .condition import VoiceCondition
from .modules import (
    AdaRMSNorm,
    IdentityPrefix,
    StyleModulation,
    StyleModulationSlot,
    StyleTrunk,
    TransformerSpec,
    VoiceConditionConfig,
    select_layers,
)

CHECKPOINT_FORMAT = "breeze-voice-conditioner-stage2-v1"

_UNSET = object()


@dataclass
class _Site:
    """One the backbone transformer stack and the slots wired into it."""

    spec: TransformerSpec
    prefix_slots: dict[int, AttentionPrefixSlot]
    style_slots: dict[int, StyleModulationSlot]


def spec_from(config, name: str) -> TransformerSpec:
    """Read one stack's shapes off its the backbone config."""
    heads = int(config.num_attention_heads)
    return TransformerSpec(
        name=name,
        num_layers=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        num_attention_heads=heads,
        num_key_value_heads=int(getattr(config, "num_key_value_heads", heads)),
        head_dim=int(getattr(config, "head_dim", config.hidden_size // heads)),
    )


class VoiceConditioner(nn.Module):
    """The Stage-2 parameters, and explicit control over where they apply."""

    def __init__(self, config: VoiceConditionConfig, backbone: TransformerSpec,
                 depth: TransformerSpec) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.specs = {"backbone": backbone, "depth": depth}

        prefix = select_layers(config.prefix_layers, backbone.num_layers)
        prefix_depth = select_layers(config.prefix_depth_layers, depth.num_layers)
        style = select_layers(config.adarms_layers, backbone.num_layers)
        style_depth = select_layers(config.adarms_depth_layers, depth.num_layers)

        self.backbone_identity = (IdentityPrefix(backbone, config, prefix)
                                  if config.identity_prefix else None)
        self.depth_identity = (IdentityPrefix(depth, config, prefix_depth)
                               if config.identity_prefix else None)
        self.style_trunk = StyleTrunk(config) if config.style_adarms else None
        self.backbone_style = (StyleModulation(backbone, config, style)
                               if config.style_adarms else None)
        self.depth_style = (StyleModulation(depth, config, style_depth)
                            if config.style_adarms else None)

        self._sites: dict[str, _Site] = {}
        self._original_norms: list[tuple[nn.Module, str, nn.Module]] = []
        self._condition: VoiceCondition | None = None
        self._depth_rows: torch.Tensor | None = None
        self._shim = False
        self._shim_handles: list = []
        self._primary = _UNSET
        self._conditional = True
        object.__setattr__(self, "_model", None)

    # ------------------------------------------------------------------ build

    @classmethod
    def for_model(cls, model: nn.Module, config: VoiceConditionConfig) -> "VoiceConditioner":
        return cls(config, spec_from(model.backbone_model.config, "backbone"),
                   spec_from(model.depth_decoder.model.config, "depth"))

    @classmethod
    def load(cls, path: Path, *, model: nn.Module | None = None) -> "VoiceConditioner":
        path = Path(path)
        payload = json.loads((path / "voice_conditioner.json").read_text())
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unexpected conditioner format {payload.get('format')!r}")
        config = VoiceConditionConfig(**payload["config"])
        specs = {}
        for key, name in (("backbone_spec", "backbone"), ("depth_spec", "depth")):
            fields = dict(payload[key])
            fields["name"] = name
            specs[name] = TransformerSpec(**fields)
        conditioner = cls(config, specs["backbone"], specs["depth"])
        state = torch.load(path / "voice_conditioner.pt", map_location="cpu",
                           weights_only=True)
        # Strict: an ignored tensor here means shipping the wrong voice.
        conditioner.load_state_dict(state, strict=True)
        return conditioner

    def attach(self, model: nn.Module) -> "VoiceConditioner":
        """Install the prefix slots and swap the modulated norms into the backbone."""
        if self._sites:
            raise RuntimeError("conditioner is already attached")
        register_attention_implementation()

        backbone_layers = model.backbone_model.layers
        depth_layers = model.depth_decoder.model.layers
        plan = (
            ("backbone", backbone_layers, self.backbone_identity, self.backbone_style),
            ("depth", depth_layers, self.depth_identity, self.depth_style),
        )
        for name, layers, identity, style in plan:
            site = _Site(self.specs[name], {}, {})
            if identity is not None:
                for index in identity.layers:
                    slot = AttentionPrefixSlot()
                    layers[index].self_attn.voice_prefix_slot = slot
                    site.prefix_slots[index] = slot
            if style is not None:
                for index in style.layers:
                    slot = StyleModulationSlot()
                    layer = layers[index]
                    for attribute, kind in (("input_layernorm", "attn"),
                                            ("post_attention_layernorm", "mlp")):
                        base = getattr(layer, attribute)
                        self._original_norms.append((layer, attribute, base))
                        setattr(layer, attribute, AdaRMSNorm(base, slot, kind))
                    site.style_slots[index] = slot
            self._sites[name] = site

        # Every config an attention module might read, not just the two stack
        # ones: the layers hold their own. Switching only the stack config
        # leaves them on stock attention, so the prefix is built and never
        # read while the norms still apply - fluent, and the wrong voice.
        configs = {}
        for stack in (model.backbone_model, model.depth_decoder.model):
            configs[id(stack.config)] = stack.config
            for layer in stack.layers:
                config = getattr(layer.self_attn, "config", None)
                if config is not None:
                    configs[id(config)] = config
        for config in configs.values():
            config._attn_implementation = ATTENTION_IMPLEMENTATION
        object.__setattr__(self, "_model", model)
        return self

    # -------------------------------------------------------------- condition

    def set_condition(self, condition: VoiceCondition | None) -> None:
        """Project the condition into every slot, once per utterance."""
        self._condition = condition
        self._primary = _UNSET
        self._conditional = True
        self._push("backbone", condition, None)
        # Indexed by an explicit identity map rather than left unindexed. Same
        # values either way, but the gather lays them out differently, which
        # selects different matmul kernels and diverges after ~5 frames.
        rows = (None if condition is None else
                torch.arange(condition.batch_size, device=condition.device))
        self._depth_rows = rows
        self._push("depth", condition, rows)

    def set_depth_rows(self, rows: torch.Tensor | None) -> None:
        """Map each depth row back to the condition row it belongs to.

        The depth decoder only runs for unfinished utterances, so its batch
        shrinks and stops lining up with the condition.
        """
        if self._condition is None:
            raise RuntimeError("set_condition must be called before set_depth_rows")
        if rows is not None and self._depth_rows is not None \
                and rows.shape == self._depth_rows.shape \
                and bool(torch.equal(rows, self._depth_rows)):
            return
        self._depth_rows = rows
        self._push("depth", self._condition, rows)

    @contextlib.contextmanager
    def branch(self, kind: str = "conditional"):
        """Run a forward pass as the conditional or unconditional branch.

        The negative pass must see no voice. Restoring on exit is the point:
        a prefix left cleared would unvoice every later frame.
        """
        if kind not in ("conditional", "unconditional"):
            raise ValueError(f"unknown branch {kind!r}")
        if kind == "conditional":
            yield
            return
        self._push("backbone", None, None)
        try:
            yield
        finally:
            self._push("backbone", self._condition, None)

    def enable_guidance_shim(self, enabled: bool = True) -> None:
        """Recognise the guidance branch by KV-cache identity.

        The one piece of runtime inference left. `branch()` is the real answer,
        but the backbone's `generate` runs both passes inside itself, so nothing
        outside can say which is starting; until that loop is owned here, the
        first cache seen after `set_condition` is taken as the conditional one.

        Correct for the current `generate`, and quietly wrong against any
        implementation that reallocates or shares caches. Opt-in for that
        reason: a caller driving both passes should use `branch()` instead.
        """
        self._shim = enabled
        if not enabled:
            for handle in self._shim_handles:
                handle.remove()
            self._shim_handles.clear()
            return
        if self._shim_handles or self._model is None:
            return
        self._shim_handles.append(
            self._model.register_forward_pre_hook(
                self._guidance_hook, with_kwargs=True))

    def _guidance_hook(self, module, args, kwargs):
        del module, args
        if not self._shim or self._condition is None:
            return
        cache = kwargs.get("past_key_values")
        key = None if cache is None else id(cache)
        if self._primary is _UNSET:
            self._primary = key
            return
        conditional = key == self._primary
        if conditional == self._conditional:
            return
        self._push("backbone", self._condition if conditional else None, None)
        self._conditional = conditional

    def reset(self) -> None:
        self._condition = None
        self._depth_rows = None
        self._primary = _UNSET
        self._conditional = True
        self._push("backbone", None, None)
        self._push("depth", None, None)

    def detach(self) -> None:
        """Put the backbone back exactly as it was found."""
        for layer, attribute, base in self._original_norms:
            setattr(layer, attribute, base)
        self._original_norms.clear()
        for site in self._sites.values():
            for slot in site.prefix_slots.values():
                slot.clear()
            for slot in site.style_slots.values():
                slot.clear()
        self._sites.clear()
        object.__setattr__(self, "_model", None)

    # ---------------------------------------------------------------- internal

    def _push(self, name: str, condition: VoiceCondition | None,
               rows: torch.Tensor | None) -> None:
        site = self._sites.get(name)
        if site is None:
            return
        identity = self.backbone_identity if name == "backbone" else self.depth_identity
        style = self.backbone_style if name == "backbone" else self.depth_style
        if condition is None:
            for slot in site.prefix_slots.values():
                slot.clear()
            for slot in site.style_slots.values():
                slot.clear()
            return
        if identity is not None and site.prefix_slots:
            identity.build(condition, site.prefix_slots, expand_index=rows)
        if style is not None and site.style_slots and self.style_trunk is not None:
            state = self.style_trunk(condition)
            valid = condition.valid_style
            if rows is not None:
                state = state[rows]
                valid = valid[rows]
            style.build(state, valid, site.style_slots)

    # ------------------------------------------------------------- persistence

    def save(self, path: Path, *, metadata: dict | None = None) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "voice_conditioner.pt")
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": self.config.to_json(),
            "backbone_spec": {k: v for k, v in asdict(self.specs["backbone"]).items()
                              if k != "name"},
            "depth_spec": {k: v for k, v in asdict(self.specs["depth"]).items()
                           if k != "name"},
        }
        if metadata:
            payload["metadata"] = metadata
        (path / "voice_conditioner.json").write_text(json.dumps(payload, indent=1) + "\n")
        return path


__all__ = ["VoiceConditioner"]

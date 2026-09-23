"""Gated identity KV-prefix attention.

The backbone layers dispatch attention through ``ALL_ATTENTION_FUNCTIONS``, so a
registered implementation is enough to prepend static identity keys/values
without editing the pinned model source.

Zero-start gating (spec 4) needs care. Setting the prefix key and value to zero
does *not* leave the pretrained model untouched: a zero key still scores a logit
of zero, so the prefix steals softmax mass from the real tokens and a zero value
then shrinks the attention output. What the gate must scale is the prefix's
contribution to the *result*, so this implements

    out = out_seq + alpha * (out_prefixed - out_seq)

which is exactly baseline attention at ``alpha = 0`` and exactly standard
KV-prefix attention at ``alpha = 1``, with a well-conditioned gradient for
``alpha`` at zero (unlike a log-space or squared gate, which starts flat).

Splitting the joint softmax over ``[prefix; sequence]`` by total mass turns that
into

    out = out_seq + alpha * m * (out_pre - out_seq)

where ``out_pre`` attends over the prefix alone and ``m = sigmoid(lse_pre -
lse_seq)`` is the share of softmax mass the prefix would take. Written this way
the ``out_seq`` term is computed by exactly the operations stock eager attention
performs, so a zero gate reproduces the pretrained model bit for bit rather than
merely closely, and the prefix work is over ``num_id_tokens`` columns only.
"""

import os

import torch
from torch import nn
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, repeat_kv

ATTENTION_IMPLEMENTATION = "soul_voice_prefix"

# The sequence half of this kernel is written out by hand only because the gate
# needs the log-sum-exp of the sequence logits, which
# ``scaled_dot_product_attention`` does not return. The fused backend under it
# does return it, so the same result is available without materialising the
# [B, H, Q, K] score matrix. Off by default: it is a different summation order,
# so it is not bit-exact against the released renders, and it only pays where
# the query is wide.
USE_FUSED_SEQUENCE = os.environ.get("SOUL_VOICE_SDPA", "0") == "1"

# FlashAttention for the same half. It returns the log-sum-exp outright, which
# is the one thing that made the hand-written branch necessary, so the gate is
# recoverable from it exactly. Only usable where the mask is nothing more than
# causal - flash takes no arbitrary bias - which during decode is every call,
# since a one-row query attends to the whole cache.
USE_FLASH_SEQUENCE = os.environ.get("SOUL_VOICE_FLASH", "0") == "1"


def _causal_only(attention_mask, query, key_states):
    """Whether this mask says anything a causal flag would not.

    A padded batch masks columns that causality would have kept, and flash has
    nowhere to put that, so those calls stay on the branch below.
    """
    if attention_mask is None:
        return True
    keys = key_states.shape[2]
    block = attention_mask[:, :, :, :keys]
    rows = query.shape[2]
    # Bottom-right alignment: row i of the query is key offset (keys - rows + i).
    positions = torch.arange(keys, device=block.device)
    limit = (keys - rows) + torch.arange(rows, device=block.device).unsqueeze(-1)
    expected = torch.where(positions <= limit, 0.0, float("-inf"))
    return bool(((block == 0) == (expected == 0)).all())


# cuDNN's fused attention for the same half. Unlike flash it takes an
# arbitrary bias, so a padded batch is no obstacle, and it returns the
# log-sum-exp outright as well.
USE_CUDNN_SEQUENCE = os.environ.get("SOUL_VOICE_CUDNN", "0") == "1"


def _cudnn_terms(query, key_states, value_states, attention_mask, scaling):
    bias = None
    if attention_mask is not None:
        bias = attention_mask[:, :, :, : key_states.shape[2]].expand(
            query.shape[0], query.shape[1], query.shape[2], key_states.shape[2]
        ).to(query.dtype).contiguous()
    out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
        query, key_states, value_states, bias, True, 0.0, False, False,
        scale=scaling)[:2]
    # Already [B, H, Q, 1], which is the shape the gate wants.
    return out, lse.to(torch.float32)


def _flash_terms(query, key_states, value_states, scaling):
    from flash_attn import flash_attn_func

    out, lse, _ = flash_attn_func(
        query.transpose(1, 2), key_states.transpose(1, 2),
        value_states.transpose(1, 2), softmax_scale=scaling, causal=True,
        return_attn_probs=True)
    return out.transpose(1, 2), lse.unsqueeze(-1).to(torch.float32)


def _sequence_terms(query, key_states, value_states, attention_mask, scaling,
                    dropout, module):
    """The attention output over the real tokens, and the mass its logits carry.

    Returns ``(out_seq, lse_seq)``. The hand-written branch is the reference and
    stays bit-exact with stock eager attention at a zero gate; the fused branch
    computes the same two quantities from the memory-efficient kernel, which
    keeps the score matrix off HBM.
    """
    live = not (module.training and dropout > 0.0)
    if USE_CUDNN_SEQUENCE and live:
        return _cudnn_terms(query, key_states, value_states, attention_mask, scaling)
    if USE_FLASH_SEQUENCE and live and _causal_only(attention_mask, query, key_states):
        return _flash_terms(query, key_states, value_states, scaling)
    if USE_FUSED_SEQUENCE and live:
        bias = None
        if attention_mask is not None:
            # The kernel wants a materialised bias whose row stride is a
            # multiple of eight. Being contiguous is not enough, since that
            # stride is then the key count itself, so the row is padded out and
            # sliced back: the slice keeps the padded stride and drops the
            # columns, which is exactly the alignment the kernel checks for.
            bias = attention_mask[:, :, :, : key_states.shape[2]].expand(
                query.shape[0], query.shape[1], query.shape[2], key_states.shape[2]
            ).to(query.dtype)
            keys = bias.shape[-1]
            padding = (-keys) % 8
            if padding:
                bias = nn.functional.pad(bias, (0, padding))[..., :keys]
            else:
                bias = bias.contiguous()
        out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
            query, key_states, value_states, bias, True, 0.0, False, scale=scaling)
        return out, lse[..., : query.shape[2]].unsqueeze(-1).to(torch.float32)

    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, : key_states.shape[-2]]
    weights = nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = nn.functional.dropout(weights, p=dropout, training=module.training)
    accumulator = torch.promote_types(logits.dtype, torch.float32)
    return (torch.matmul(weights, value_states),
            torch.logsumexp(logits.to(accumulator), dim=-1, keepdim=True))

# Bias applied to a dropped sample's prefix logits. Deliberately finite: the
# dtype minimum makes the prefix-only softmax degenerate to NaN, which then
# survives multiplication by a zero gate.
MASKED_PREFIX_LOGIT = -1e9


class AttentionPrefixSlot:
    """Per-layer handoff between the conditioner and the attention kernel.

    Deliberately a plain object rather than a buffer or module attribute: the
    tensors change every forward, must not enter ``state_dict``, and must stay
    alive into backward when gradient checkpointing recomputes the layer.
    """

    __slots__ = ("bias", "gate", "key", "value")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.key = None
        self.value = None
        self.gate = None
        self.bias = None

    def set(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        self.key = key
        self.value = value
        self.gate = gate
        self.bias = bias

    @property
    def active(self) -> bool:
        return self.key is not None


def voice_prefix_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    slot: AttentionPrefixSlot | None = getattr(module, "voice_prefix_slot", None)
    if slot is None or not slot.active:
        return eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )

    groups = module.num_key_value_groups
    key_states = repeat_kv(key, groups)
    value_states = repeat_kv(value, groups)
    # The prefix is projected by fp32 modules; under autocast it arrives in the
    # activation dtype already, but plain bf16 inference needs the cast.
    prefix_key = repeat_kv(slot.key.to(query.dtype), groups)
    prefix_value = repeat_kv(slot.value.to(query.dtype), groups)

    prefix_logits = torch.matmul(query, prefix_key.transpose(2, 3)) * scaling
    if slot.bias is not None:
        prefix_logits = prefix_logits + slot.bias.to(prefix_logits.dtype)

    # Identical to eager_attention_forward, so a zero gate is bit-exact.
    baseline, sequence_lse = _sequence_terms(
        query, key_states, value_states, attention_mask, scaling, dropout, module
    )

    prefix_weights = nn.functional.softmax(
        prefix_logits, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    prefix_context = torch.matmul(prefix_weights, prefix_value)
    # Accumulate the mass ratio at fp32 or better; the logits themselves are
    # usually bf16, where the exponent difference would be far too coarse.
    accumulator = torch.promote_types(prefix_logits.dtype, torch.float32)
    prefix_mass = torch.sigmoid(
        torch.logsumexp(prefix_logits.to(accumulator), dim=-1, keepdim=True)
        - sequence_lse.to(accumulator)
    ).to(query.dtype)

    gate = slot.gate.to(query.dtype) * prefix_mass
    attn_output = baseline + gate * (prefix_context - baseline)

    return attn_output.transpose(1, 2).contiguous(), None


def register_attention_implementation() -> str:
    """Make the implementation selectable via ``config._attn_implementation``.

    The mask entry matters as much as the attention entry: ``create_causal_mask``
    silently returns ``None`` for implementations it does not know, which would
    drop causal masking entirely.
    """
    if ATTENTION_IMPLEMENTATION not in ALL_ATTENTION_FUNCTIONS._global_mapping:
        ALL_ATTENTION_FUNCTIONS.register(
            ATTENTION_IMPLEMENTATION, voice_prefix_attention_forward
        )
    if ATTENTION_IMPLEMENTATION not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, eager_mask)
    return ATTENTION_IMPLEMENTATION


__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "AttentionPrefixSlot",
    "register_attention_implementation",
    "voice_prefix_attention_forward",
]

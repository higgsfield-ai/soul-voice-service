"""A cached depth decoder, as a drop-in for the one the backbone ships.

Soul Voice emits sixteen codebooks per frame: the backbone produces codebook 0,
then a small depth decoder fills 1 to 15 one step at a time. That stack runs
fifteen times per backbone step, twice more under guidance, so it is where the
frame budget goes. The shipped version runs those steps with `use_cache=False`
and feeds the whole frame back in each time, recomputing attention over
codebooks it already attended to.

Caching them gives the same tokens because the only two things that vary with
a cache - the codebook offset and the hidden-state injection - are both driven
by `cache_position`, which the backbone derives from the cache length. One token at
position t with a cache of length t reproduces the uncached offset. The hidden
state goes in on the first step only; the injection is not guarded by
position, so passing it again overwrites a real codebook embedding.

Off by default, because it measured *slower*: 35.7s against 32.3s for two
prompts. The sequence it avoids recomputing is at most sixteen long, so the
saved recompute costs less than the fifteen extra kernel launches. A longer
frame, a larger batch or a CUDA-graph capture would each flip that, which is
why it is kept and tested rather than deleted.

It is also token-identical under greedy decoding but not under sampling: a
one-token query selects different kernels than a sixteen-token one, and the
last bf16 bit is enough to change what `multinomial` draws. Equally good
clips, different clips - so it cannot reproduce a released render.
"""

import torch
from torch import nn


def sample_codebook(logits: torch.Tensor, *, do_sample: bool, temperature: float,
                    top_k: int | None, top_p: float | None) -> torch.Tensor:
    """The shipped sampling order, preserved exactly.

    Top-k applies to probabilities after the softmax and renormalises, not to
    logits before it. The two differ, and the released renders used this one.
    """
    if temperature is not None and temperature != 1.0:
        logits = logits / temperature
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probabilities = nn.functional.softmax(logits, dim=-1)
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(probabilities, min(top_k, probabilities.size(-1)))
        probabilities = torch.zeros_like(probabilities).scatter_(-1, indices, values)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    if top_p is not None and top_p < 1.0:
        ordered, order = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(ordered, dim=-1)
        drop = cumulative > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = 0
        probabilities = probabilities.masked_fill(
            drop.scatter(-1, order, drop), 0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1)


def cached_depth_generate(
    model,
    depth_decoder_input_ids: torch.LongTensor,
    cond_backbone_hidden_state: torch.FloatTensor,
    uncond_backbone_hidden_state: torch.FloatTensor,
    cfg_scale: float,
) -> torch.LongTensor:
    """Codebooks 1..15 under guidance, with a KV cache across the steps.

    Signature matches `BreezeGenerationMixin._depth_decoder_generate_with_cfg`
    so it can be bound in its place.
    """
    decoder = model.depth_decoder
    settings = decoder.generation_config
    sequences = depth_decoder_input_ids            # [B, 2]: placeholder, codebook 0
    # Left as None so the model builds its own cache on the first step: the
    # cache has to be constructed against this config to lay its layers out
    # correctly, and a bare DynamicCache() silently is not the same object.
    caches: list = [None, None]
    hidden = (cond_backbone_hidden_state, uncond_backbone_hidden_state)

    for step in range(model.config.num_codebooks - 1):
        # The cache holds everything before this step, so only the newest
        # column is fed; at step 0 that is the placeholder and codebook 0.
        feed = sequences if step == 0 else sequences[:, -1:]
        # Explicit, and this is the subtle part: each codebook has its own
        # output head, chosen by position. Without `cache_position` the head
        # falls back to `arange(len(feed))` (models/breeze.py line 615), right
        # only for a full-frame feed. An incremental one then reads head 0
        # every step, which still produces tokens, just the wrong ones.
        start = 0 if step == 0 else step + 1
        position = torch.arange(start, start + feed.shape[1], device=feed.device)
        branch = []
        for index, state in enumerate(hidden):
            out = decoder(
                input_ids=feed,
                cache_position=position,
                # First step only: the injection overwrites column 0 whatever
                # the position, so repeating it would clobber a codebook.
                backbone_last_hidden_state=state if step == 0 else None,
                past_key_values=caches[index],
                use_cache=True,
                # The default drops column 0, because in a full-frame feed that
                # column is the backbone hidden state rather than a codebook.
                # An incremental feed has no such column, and the default would
                # leave nothing to take logits from.
                logits_to_keep=0 if step == 0 else 1,
                return_dict=True,
            )
            caches[index] = out.past_key_values
            branch.append(out.logits[:, -1, :].float())

        logits = branch[1] + cfg_scale * (branch[0] - branch[1])
        model._mask_reserved_codec_logits(logits)
        nxt = sample_codebook(
            logits, do_sample=settings.do_sample, temperature=settings.temperature,
            top_k=settings.top_k, top_p=settings.top_p)
        sequences = torch.cat([sequences, nxt], dim=-1)
    return sequences


def fused_depth_generate(
    model,
    depth_decoder_input_ids: torch.LongTensor,
    cond_backbone_hidden_state: torch.FloatTensor,
    uncond_backbone_hidden_state: torch.FloatTensor,
    cfg_scale: float,
    pad_frame: bool = False,
) -> torch.LongTensor:
    """Both guidance branches in one pass of twice the batch.

    The shipped loop traverses the twelve depth layers once per branch, thirty
    times a frame. The two branches differ only in the backbone hidden state
    they are handed - the voice conditioning on this stack is identical across
    them - so they stack into one forward and the count halves.

    This is where the frame budget is. The depth stack is small enough that a
    step costs about what its kernel launches cost, so the wider matmul is very
    nearly free and halving the traversals is very nearly a halving.

    `pad_frame` feeds the full frame width every step instead of the sequence
    so far, reading the logits back at the position that was actually filled.
    The stack is causal, so the columns past that position cannot reach it and
    the result is unchanged - what changes is that every step now has the same
    shape, which is what makes the loop capturable. It costs arithmetic that a
    launch-bound stage was not spending anyway.
    """
    decoder = model.depth_decoder
    settings = decoder.generation_config
    sequences = depth_decoder_input_ids                    # [B, 2]
    batch = sequences.shape[0]
    hidden = torch.cat([cond_backbone_hidden_state, uncond_backbone_hidden_state])
    frame = None
    if pad_frame:
        frame = sequences.new_zeros((2 * batch, model.config.num_codebooks))

    for _ in range(model.config.num_codebooks - 1):
        filled = sequences.shape[1]
        both = torch.cat([sequences, sequences])
        if frame is not None:
            frame[:, :filled] = both
        out = decoder(
            input_ids=both if frame is None else frame,
            # Passed every step, as the uncached path must: the injection
            # overwrites column 0, which is fed again each time.
            backbone_last_hidden_state=hidden,
            use_cache=False,
            return_dict=True,
            # Every position, so the one to read is chosen out here rather than
            # by a keep count that would change with the step and put the shape
            # back in the graph. Each position carries its own codebook head,
            # so the column below is still the right head.
            **({"logits_to_keep": 0} if frame is not None else {}),
        )
        # Asking for every position shifts the column: the head predicts the
        # next codebook, so its row j answers for input position j + 1 and the
        # position just filled is at j = filled - 2.
        logits = out.logits[:, -1 if frame is None else filled - 2, :].float()
        conditional, unconditional = logits[:batch], logits[batch:]
        blended = unconditional + cfg_scale * (conditional - unconditional)
        model._mask_reserved_codec_logits(blended)
        sequences = torch.cat([sequences, sample_codebook(
            blended, do_sample=settings.do_sample, temperature=settings.temperature,
            top_k=settings.top_k, top_p=settings.top_p)], dim=-1)
    return sequences


class RowTracker:
    """Keeps the conditioner aligned with the depth decoder's shrinking batch.

    the backbone drops finished utterances before the depth decoder runs, so its rows
    stop lining up with the condition and the audio comes out in another row's
    voice - the shapes broadcast, so nothing raises. The active mask is a local
    of the backbone's loop, so rows are matched on hidden states as the research
    conditioner did, except that an ambiguous match raises here.
    """

    def __init__(self, conditioner) -> None:
        self.conditioner = conditioner
        self.hidden: torch.Tensor | None = None
        self.handle = None

    def attach(self, model) -> "RowTracker":
        def capture(module, args, kwargs, output):
            del module, args, kwargs
            states = getattr(output, "hidden_states", None)
            # Only the conditional branch: it is the one whose rows the
            # condition is indexed by.
            if states and getattr(self.conditioner, "_conditional", True):
                self.hidden = states[-1][:, -1, :].detach()
            return output

        self.handle = model.register_forward_hook(capture, with_kwargs=True)
        return self

    def align(self, active: torch.Tensor, *, repeat: int = 1) -> None:
        """Tell the conditioner which condition row each active row is.

        `repeat` tiles the mapping for a fused pass, whose batch is the active
        rows once per guidance branch.
        """
        full = self.hidden
        rows = None
        if full is not None and active.shape[0] != full.shape[0]:
            hits = (full.unsqueeze(0) == active.unsqueeze(1).to(full.dtype)).all(dim=-1)
            counts = hits.sum(dim=1)
            if not bool((counts == 1).all()):
                raise RuntimeError(
                    "cannot map the depth decoder's rows back to the condition: "
                    f"{int((counts == 0).sum())} rows matched nothing and "
                    f"{int((counts > 1).sum())} matched more than one")
            rows = hits.float().argmax(dim=1)
        if repeat > 1:
            if rows is None:
                rows = torch.arange(active.shape[0], device=active.device)
            rows = rows.repeat(repeat)
        self.conditioner.set_depth_rows(rows)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


DEPTH_MODES = ("fused", "cached", "shipped")


def install(model, tracker: RowTracker | None = None, *, mode: str = "fused",
            pad_frame: bool = False) -> None:
    """Wrap the depth decoder's guided generation on a loaded model.

    `fused` runs both guidance branches in one pass and is the default.
    `shipped` is the released loop, kept because parity is measured against it.
    `cached` is the incremental one, kept because its verdict belongs to the
    current shapes. Realignment happens whichever runs underneath, so the three
    are comparable. Bound rather than edited in, so the third-party source
    stays a pristine copy of what was released.
    """
    import types

    if mode not in DEPTH_MODES:
        raise ValueError(f"depth mode must be one of {DEPTH_MODES}, got {mode!r}")
    shipped = model._depth_decoder_generate_with_cfg

    def guided(self, depth_decoder_input_ids, cond_backbone_hidden_state,
               uncond_backbone_hidden_state, cfg_scale):
        if tracker is not None:
            tracker.align(cond_backbone_hidden_state,
                          repeat=2 if mode == "fused" else 1)
        if mode == "fused":
            return fused_depth_generate(
                self, depth_decoder_input_ids, cond_backbone_hidden_state,
                uncond_backbone_hidden_state, cfg_scale, pad_frame=pad_frame)
        if mode == "cached":
            return cached_depth_generate(
                self, depth_decoder_input_ids, cond_backbone_hidden_state,
                uncond_backbone_hidden_state, cfg_scale)
        return shipped(
            depth_decoder_input_ids=depth_decoder_input_ids,
            cond_backbone_hidden_state=cond_backbone_hidden_state,
            uncond_backbone_hidden_state=uncond_backbone_hidden_state,
            cfg_scale=cfg_scale)

    model._depth_decoder_generate_with_cfg = types.MethodType(guided, model)


__all__ = ["DEPTH_MODES", "RowTracker", "cached_depth_generate",
           "fused_depth_generate", "install", "sample_codebook"]

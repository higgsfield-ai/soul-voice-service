# soul-voice

Inference stack for **Soul Voice**, split out of the training repository so a
serving team can run it without the trainers, the datasets or the RL code.

Soul Voice is a voice-conditioned text-to-speech model: a released Breeze-TTS 2
checkpoint for the backbone and codec, plus ~180M of conditioning trained here
that decides who is speaking and how. This package is the half that serves it.
Where the text below says *the backbone*, it means the upstream half, whose
source and symbols keep their original names.

Verified against the research pipeline: on the bit-exact path
(`depth="shipped"`) the same bundle and seed produce **bit-identical** audio in
all three modes, on both the RAFT round-0 and GRPO round-1 checkpoints. The
faster default is token-identical to that path under greedy decoding.
`parity_check.py` asserts both.

```python
from soul_voice import Request, VoiceConsumer

consumer = VoiceConsumer.load(
    "checkpoints/round1",
    source_dir="third_party/breeze-tts",
    device="cuda:0",
)
print(consumer.verify())                      # refuse to start if this is unhappy

audio, = consumer.synthesize([Request(
    text="We go live in thirty seconds.",
    instruction="An urgent young female voice, clipped and low.",
    seed=4242,
)])                                           # float32 mono at 24 kHz
```

---



## Install

```bash
poetry install
```

`torch>=2.4` resolves to whatever wheel PyPI offers; install the CUDA build
you need first if it matters. Developed against 2.9.1+cu130.

The backbone's source (`models.breeze`, `breeze_infer`) is not on PyPI. Point
`VoiceConsumer.load(source_dir=...)` at a checkout of Breeze-TTS 2 and the
loader puts it on `sys.path`. It stays an unmodified copy of what was
released: `decode.install()` binds onto a loaded model rather than editing it.

---



## Voice conditioning

The ~180M this package exists to wire up correctly:


| Component             | Params | Role                                              |
| --------------------- | ------ | ------------------------------------------------- |
| Voice conditioner     | 116M   | Injects the voice into all 40 transformer layers  |
| Voice encoder         | 47M    | Reference recording to latents (clone, direction) |
| Instruction projector | 17M    | Written description to latents (design)           |


All three modes end in a `VoiceCondition`: `id_tokens` `[B,8,512]` for *who is
speaking*, `style_tokens` `[B,4,512]` for *how*, and per-row validity flags.
Where those two halves come from is the whole difference between the modes:


| Mode        | Identity (`id_tokens`) | Delivery (`style_tokens`)     | Needs                   |
| ----------- | ---------------------- | ----------------------------- | ----------------------- |
| `design`    | the description        | the description               | instruction             |
| `clone`     | the recording          | the recording                 | instruction + reference |
| `direction` | the recording          | blended, by `style_mix_alpha` | instruction + reference |




### How the voice reaches the model

Two paths, both applied to every backbone and depth layer:

**Identity** becomes a static key/value prefix. Note that this is *not*
ordinary prefix attention over `[prefix; sequence]`. Attention over the real
tokens is computed exactly as stock attention does, attention over the prefix
is computed separately, and the two are blended:

```
out = out_seq + gate * m * (out_prefix - out_seq)
m   = sigmoid(lse_prefix - lse_seq)
```

`m` is the share of softmax mass the prefix would have taken in a joint
softmax. The shape matters for a reason worth keeping: at `gate = 0` this is
bit-exact stock attention, so training could start from the pretrained model
without perturbing it. Reimplementing it as plain KV concatenation would be a
different function.

**Style** becomes FiLM modulation on the pre-norms: each selected `RMSNorm` is
replaced by `RMSNorm(x) * (1 + gamma) + beta`, where gamma and beta come from a
shared trunk over the style tokens. The replacement adopts the pretrained
`weight` by reference, so the backbone's `state_dict` keys are unchanged.

Both are projected **once per utterance**, not per token.

---



## Drawing voices

`seed` and `style_mix_alpha` are read per request, but they are applied once
per batch, so requests are grouped by `(mode, seed, style_mix_alpha)` before
batching. Two requests that differ in any of the three are rendered in
separate passes. A seed is reproducible: the same request renders
bit-identical audio every time.

### design - inventing a voice

The instruction projector turns the description into latents. By default it
emits a single averaged latent, so **a description maps to one settled
voice**: render it today and next week and the same person reads it, with
only token sampling moving between takes. That is what most callers want from
a description, so it is the default.

To cast *different* people from one description, attach a **voicebook** and
ask for a draw:

```python
consumer = VoiceConsumer.load(bundle, source_dir=..., voicebook="voicebook.pt")
voices = consumer.synthesize(requests, sample_voices=True)
```

Each row then draws its own real speaker prototype from the bank. Both parts
are required: without a voicebook there is nothing to draw from and the flag
does nothing, and `verify()` prints which case you are in.

Two ways to get *N* different voices for one description, both measured on the
round-1 bundle with a 1024-prototype voicebook, as mean pairwise identity
cosine across the renders (lower is more distinct):

```python
# N rows in one call: each row draws its own prototype.   agreement 0.396
voices = consumer.synthesize([
    Request(text=text, instruction=instruction) for _ in range(4)],
    sample_voices=True)

# N calls, one seed each: each call reseeds, then draws.  agreement 0.614
voices = [consumer.synthesize(
    [Request(text=text, instruction=instruction, seed=seed)],
    sample_voices=True)[0] for seed in range(4)]
```

The first is both cheaper and more varied, so prefer it when you want a cast
of speakers. Use the second when each voice has to be recoverable from its
seed later. `voice_top_k` and `voice_temperature` on `synthesize` sharpen or
widen the draw.

### clone - copying a voice

Identity and delivery both come from the recording; `style_mix_alpha` is
ignored.

```python
audio, = consumer.synthesize([Request(
    text=text, instruction=instruction,
    mode="clone", reference="speaker.wav", seed=4242)])
```

`reference.py` reads the file to mono 24 kHz, applies
`manifest["reference_highpass_hz"]`, and takes an **8-second crop**. The seed
picks the crop when the recording is longer than that, so different seeds hear
different seconds of the same speaker; for a recording of 8 seconds or less
the whole file is used and the seed only changes the take. Either way the same
seed reproduces exactly.

For several takes of one voice, vary the seed across calls. The identity holds
because it comes from the recording, not from a draw.

### direction - a recording, delivered differently

Identity stays with the recording while delivery moves toward the description,
by `style_mix_alpha`:

- `0.0` keeps the recording's own delivery, which is `clone`.
- `1.0` is a full instruction override, the default.
- values between blend the two style latents linearly.

```python
takes = consumer.synthesize([
    Request(text=text, instruction="Exhausted, trailing off.",
            mode="direction", reference="speaker.wav",
            seed=4242, style_mix_alpha=alpha)
    for alpha in (0.25, 0.5, 1.0)])
```

Because the alpha is part of the batching key, a sweep like this renders as
three passes rather than one - correct, and three times the work. Sweeping
alpha is a tuning activity, not a serving one.

---



## Architecture

Three and a half billion parameters in six pieces, of which the RL rounds
train three small ones. Sizes are what `verify()` counts.

```
        "the line to speak"          design: an instruction     clone: a .wav
                 |                            |                       |
                 v                            v                       v
         text tokenizer               InstructionProjector      ReferenceFrontend
                 |                        16.6M                 resample, high-pass,
        +--------+--------+                  |                  crop -> codec frames
        v                 v            PrototypeVoicebook              |
 embed_text_tokens   T5Gemma2Text      one Gumbel draw per row         v
      536.9M         Encoder 999.9M          |                  VoiceEncoder 47.2M
 the whole token     contextual              |                         |
 sequence, incl.     features for            +-----------+-------------+
 audio specials      the text spans                      |
        |                 |                              v
        |                 v                    8 identity + 4 style latents
        |        text_encoder_proj 2.4M                  |
        |                 |                              v
        +--------+--------+                    VoiceConditioner 116.2M
                 |                          +-------------+-------------+
     encoder features are                   v                           v
     substituted into the         8 keys/values prefixed      style -> AdaRMSNorm
     embedded sequence            to attention, gated         scale and shift
                 |                          |                           |
                 v                          v                           v
      +----------------------------------------------------------------------+
      |  backbone   BreezeBackboneAdapter                            1476.6M  |
      |  28 layers, d=2048, 16 heads / 8 kv, ffn 6144                         |
      +----------------------------------------------------------------------+
                 |  one hidden state per frame
                 v
           lm_head 4.2M  ->  codebook 0            vocab 2051, pad 2050
                 |
                 v
      +----------------------------------------------------------------------+
      |  depth decoder   BreezeDepthDecoderForCausalLM                434.3M  |
      |  12 layers, d=1024, 8 heads / 2 kv, ffn 8192                          |
      |  15 sequential steps -> codebooks 1..15   (same conditioning applied) |
      +----------------------------------------------------------------------+
                 |  16 codes for this frame
                 v
           Mimi codec 79.3M / audio_tokenizer.decode
                 |
                 v
        waveform at 24 kHz - 1920 samples, so 80 ms per frame
```

A frame is one pass of the backbone and fifteen of the depth decoder, and
**every one of those runs twice**, once conditioned and once not, blended at
the logits by the guidance scale. That is thirty-two passes for 80ms of audio,
which is where `decode.py` and `accelerate.py` spend their effort: the two
guidance branches are fused into one wider pass, and the depth loop is
captured in a CUDA graph.

Sequence length follows from the frame rate: the backbone sees the prompt plus
one position per frame, so it ends at `prompt + 12.5 x seconds` and averages
about half that. The depth decoder is 16 positions whatever the duration.

```
audio   prompt   frames   final context
1.92s       11       24              35
5.36s       25       67              92
8.80s       40      110             150
```

Attention is therefore never where the time goes, and `max_new_tokens=1024` is
a ceiling of 81.9 seconds. Note also that a second of audio is 200 codebook
tokens (12.5 frames x 16), so token throughput and sequence length are very
different numbers here.

The conditioning reaches every layer of both stacks - 28 and 12 - rather than
being prepended once at the input. Identity arrives as eight extra keys and
values that attention may attend to, behind a zero-initialised gate; style
arrives as a scale and shift on the norms. `attention.py` explains why the
gate is written the way it is.

Of this, the RL rounds train only the conditioner, the projector and (in RAFT)
the voice encoder - 180M of the 3466M. The backbone, the text encoder and the
codec are frozen throughout, which is why one copy of each serves every
checkpoint.

---



## What this package changed, and why

The research stack worked by observing the model at runtime. That is fine in a
notebook and wrong in production, because both inferences fail *silently* -
they yield audio in the wrong voice, not an exception.

**Guidance branches.** Classifier-free guidance runs an instructed, voiced pass
against a bare one. The research conditioner told them apart by remembering the
identity of the first KV cache object it saw. `branch()` replaces this for
callers that drive the two passes themselves. The backbone's shipped `generate` runs
both inside itself, so until that loop is owned here too, `enable_guidance_shim()`
keeps the old trick - opt-in, isolated, and documented as the weakest link.

**Depth rows.** Finished utterances are dropped before the depth decoder, so
its batch stops lining up with the condition. The research conditioner
recovered the mapping by comparing hidden states and continued if the match was
ambiguous. `RowTracker` keeps the comparison - the active mask is a local of
the backbone's loop and there is nothing else to match on - but **raises** when the
match is ambiguous or missing.

**Loading.** `verify()` accounts for every tensor and checks that the
conditioning actually reached the model. `load(strict=True)` refuses to start
otherwise.

---



## Things that fail silently

Each of these produces plausible audio in the wrong voice. Three of them were
live bugs found while building this.

**Per-layer attention dispatch.** The layers hold their *own* config objects,
separate from the stack config. Setting `_attn_implementation` on
`backbone_model.config` alone leaves all 28 layers dispatching through stock
attention: the identity prefix is built, stored, and never read, while the
modulated norms still apply. The output is fluent, confident, and not the
requested voice. `verify()` now asserts this per layer.

**The codebook head's position.** Each codebook has its own output head, chosen
by position from `cache_position`. Given none, it falls back to
`arange(len(input))` - correct only when the whole frame is fed at once, which
is what the uncached path always does. Feed one token without saying where it
sits and every step reads head 0. Any incremental depth decoding must pass
`cache_position` explicitly.

**The missing voicebook.** The instruction projector can draw from a bank of
real speaker prototypes, which is what makes two renders of one description
sound like two different people. **No bundle in this repository ships**
`voicebook.pt`, so the projector falls back to a free-form latent and
produces an averaged voice. Pass `voicebook=` explicitly; `verify()` reports
its absence. Measured identity agreement across five renders of one prompt:
0.917 without it (one voice, five readings), 0.534 with it (five voices).
See [Drawing voices](#drawing-voices) for how to sample from it.

**Reference filtering.** `manifest["reference_highpass_hz"]` is the filter the
checkpoint was trained with, and `reference.py` applies it on the way in.
Bypassing that loader and feeding unfiltered audio asks a model trained on
filtered references to map a humming recording onto a hum-free one, and
identity pays for the mismatch. The bundles here carry `0.0`, so the filter is
a no-op for them and live for anything retrained with it.

---



## Performance

`bench.py` reproduces everything below on one B300. Every row of a batch is
the same request, so audio seconds scale exactly with the batch; real traffic
has ragged lengths and a batch runs until its longest row finishes, so read
these as an upper bound on what batching buys.

```
                 ms per frame                        realtime factor
batch    shipped    fused   +graphs   total      shipped  fused  +graphs
    1      543.6    282.8      96.2   5.65x         0.15   0.28     0.83
    2      566.4    291.4      94.6   5.99x         0.24   0.50     1.46
    4      573.0    296.0     109.6   5.23x         0.53   1.00     2.84
    8      576.2    298.6     103.8   5.55x         0.88   1.79     5.54
```

A frame is 1920 samples, so **80ms of audio and 12.5 frames a second**, and
the column is the cost of advancing the whole batch one frame - which is why
it barely moves with the batch while the realtime factor climbs. One clip
alone went from 24.5s of wall clock to 4.1s; a batch of eight, from 35.7s to
5.7s.

### Where the time was

Measuring the two stages where they are called and leaving the backbone as the
remainder, the answer was not close:

```
batch     backbone       depth        codec      ms/frame
    1    3.4s   14%   21.0s   86%   0.0s   0%       543.6
    8    4.8s   13%   30.6s   86%   0.4s   1%       576.2
```

**The depth decoder was 86% of the render**, and the per-frame cost barely
moved between batch 1 and batch 8 - 544ms against 576ms for eight times the
work. A stage that costs the same whether it computes one row or eight is not
computing; it is waiting on kernel launches. Timing single passes against what
the memory bus allows says the same thing from the other side:

```
                    measured   weights   roofline at ~8 TB/s
backbone step        21.5 ms   2.95 GB              0.37 ms
depth pass (2 rows)   8.0 ms   0.87 GB              0.11 ms
```

Nearly two orders of magnitude off. At batch 1 a decode step should be a
streaming read of the weights, so essentially all of that is per-kernel
overhead. The depth stack is twelve small layers over at most sixteen tokens,
and the shipped loop traverses it **thirty times a frame**: fifteen codebooks,
twice each, because the two guidance branches are separate forward passes.

The context it runs over is tiny - about 150 positions for a nine-second clip,
and never more than sixteen in the depth decoder (see Architecture). That is
why no attention kernel moves the needle below: the cost is weights and
launches, not sequence.

### Fusing the guidance branches

The two branches are handed different backbone hidden states, but the voice
conditioning on this stack is identical across them - `branch()` clears the
backbone prefix, never the depth one. So they stack into one pass of twice the
batch, and thirty traversals a frame become fifteen. The wider matmul is very
nearly free precisely because the stage was never compute-bound. Worth 1.9x,
and it is the default (`fused_depth_generate` in `decode.py`).

### Capturing the loop

What is left after fusing is still launches, so the fix is to stop issuing
them: `torch.compile(mode="reduce-overhead")` captures the depth decoder into
CUDA graphs. Worth another 3x, and it takes the depth stack from 86% of the
render to 18%.

```python
consumer = VoiceConsumer.load(bundle, source_dir=..., compile=True)
```

Off by default only because compilation costs a few minutes on the first
request. Any process that will serve more than a handful should turn it on.
Two things had to be arranged for it to work at all, both in `accelerate.py`
and `decode.py`:

**The frame is padded to its full width.** A growing sequence is a new shape
every step, so `dynamic=False` recompiles fifteen times a frame and dynamo
gives up - and what it does when it gives up is run eager for the rest of the
process, silently, so the capture looks installed and buys nothing. Feeding
all sixteen columns every step and reading back the one just filled makes the
shape constant. The stack is causal, so the columns past that one cannot reach
it; `parity_check.py` asserts exactly that by rendering the same position
under three different fills and requiring the logits to match to the bit.

**Triton's assembler needed replacing.** It ships its own `ptxas`, and on this
GPU that copy rejects the architecture outright (`sm_103a` is not defined for
CUDA 12.8). `repair_ptxas` finds a newer one beside it and points Triton
there. Without this, compilation raises mid-request.

### The cost, and how to decline it

Fused is **token-identical to the shipped loop under greedy decoding** - check
2 of `parity_check.py` asserts that, which is why the fast path has a test at
all. Neither fused nor the padded frame is bit-identical under sampling: a
wider matmul picks different kernels, the last bit of bf16 moves, and
`multinomial` draws differently a few frames in. Equally good clips, different
clips - the renders diverge in length, not in quality.

Where reproducing a released render matters, `VoiceConsumer.load(..., depth="shipped", compile=False)` restores the bit-exact path at the original
speed, and that is what checks 3 and 4 of the parity script run.

### Attention backends, measured

All four compute the same thing; the gate needs the log-sum-exp of the
sequence logits, and each of these can return it one way or another. None of
them is the lever, because the context is ~150 keys.

```
                          ms per frame            with graph capture
backend                batch 1   batch 8        batch 1   batch 8
hand-written (default)   282.8     298.6           96.2     103.8
cuDNN fused              305.9     344.6           84.7     107.6
SDPA (efficient)             -         -           93.2     110.8
flash-attn 2.8.3         357.0     342.0          257.8     378.4
```

**flash-attn is installable here** - the official
`flash_attn-2.8.3+cu13torch2.9` wheel runs on this B300's `sm_103` and agrees
with the reference to bf16 tolerance - but it is the worst option in both
columns, and it is a disaster under capture: depth goes from 1.0s back to
8.1s, because the custom op breaks the graph. Graph capture is worth far more
than any kernel choice, so anything that costs it loses by default.

cuDNN (`SOUL_VOICE_CUDNN=1`) is the only alternative that survives capture
intact, and it is worth roughly 10% at batch 1 and slightly negative at batch
8 - close enough to the run-to-run spread that it is not the default. It is
the one to reach for if the prompt side ever grows, since it also takes an
arbitrary bias and so handles a padded batch, which flash cannot.

### Also measured and rejected

- **The backbone KV cache.** `load()` inherits `config.use_cache = False` from
the trainer, which looks alarming for autoregressive decode. `generate`
never reads it - forcing it true changed nothing.
- **Batching the codec.** It decodes one sample at a time in a Python loop
(`generation_breeze.py:1237`, whose own comment says it should be batched),
but it is 1-5% of the render. Not worth the risk.
- **The cached depth decoder** (`depth="cached"`) avoids recomputing a sequence
at most sixteen long and costs more kernel launches to do it - the wrong
trade for a launch-bound stage. Kept and tested because that verdict belongs
to the current shapes.


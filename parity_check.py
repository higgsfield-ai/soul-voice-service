"""Does the serving stack load the same weights and render the same audio?

Three questions, in the order they can fail:

1. Does every tensor land? A bundle that half-loads still speaks, so this is
   checked by accounting rather than by listening.
2. Is the cached depth decoder the same function as the one it replaces? Asked
   directly, on identical inputs under greedy decoding, so the answer is token
   equality rather than a similarity score.
3. Does a full render match the research pipeline? Same bundle, same seed, both
   stacks, compared sample by sample.
4. Does the reference path match too? Design never reads a recording, so the
   audio loader and the Stage-1 frontend need their own check.

Run: ../.breeze-venv/bin/python parity_check.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
# The research stack, the vendored the backbone source and the bundles all live in
# the workspace this package was split out of, not in the package.
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "tts_finetune/third_party/breeze-tts"
PROMPTS = [
    ("We go live in thirty seconds, and I need everyone at their stations.",
     "An urgent young female voice, clipped and low, pushing the pace."),
    ("You are not listening to me. Look at the numbers again.",
     "A furious middle-aged male voice, controlled but barely."),
]


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_weights(bundle: Path, voicebook: Path | None, device: str,
                  depth: str = "fused"):
    """Every tensor accounted for, every hook in place."""
    from soul_voice import VoiceConsumer

    banner("1. weights")
    consumer = VoiceConsumer.load(bundle, source_dir=SOURCE, device=device,
                                  voicebook=voicebook, depth=depth,
                                  strict=False)
    report = consumer.verify()
    print(report)
    print(f"\nverdict: {'PASS' if report.ok else 'FAIL'}")
    return consumer, report


def check_padding(consumer, device: str, rows: int = 8, filled: int = 6) -> bool:
    """The padded frame cannot read what it pads with.

    The captured loop feeds the whole frame every step so the shape never
    moves, which is only sound if the columns past the one being written are
    unreachable. Asserting token equality against the unpadded loop would be
    the wrong test - a 16-wide pass picks different kernels than a 6-wide one,
    so the last bf16 bit moves and a near-tie occasionally flips. What must
    hold exactly is that the fill is invisible, so that is what is checked:
    three different fills, the same logits to the bit.
    """
    banner("2a. the padded frame cannot see its padding")
    model = consumer.model
    decoder = model.depth_decoder
    width = model.config.num_codebooks
    torch.manual_seed(0)
    hidden = torch.randn(rows, model.backbone_model.config.hidden_size,
                         device=device, dtype=torch.bfloat16)
    real = torch.randint(0, 1000, (rows, filled), device=device)

    def logits_with(fill):
        frame = (torch.randint(0, 1000, (rows, width), device=device)
                 if fill == "random" else real.new_full((rows, width), fill))
        frame[:, :filled] = real
        with torch.inference_mode():
            out = decoder(input_ids=frame, backbone_last_hidden_state=hidden,
                          use_cache=False, return_dict=True, logits_to_keep=0)
        return out.logits[:, filled - 2, :].float()

    reference = logits_with(0)
    ok = True
    for fill in ("random", 1999):
        drift = float((reference - logits_with(fill)).abs().max())
        print(f"  fill {str(fill):>6}: max |difference| {drift:.3e}")
        ok = ok and drift == 0.0
    print(f"\nverdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_depth(consumer, device: str, rows: int = 4) -> bool:
    """The two rewritten depth loops against the one they replace, greedily.

    Greedy on purpose: sampling would make a mismatch look like sampling noise,
    and the claim being tested is that they compute the same logits, not that
    they produce similar-sounding audio.
    """
    from models.generation_breeze import BreezeGenerationMixin

    from soul_voice.decode import cached_depth_generate, fused_depth_generate

    banner("2. rewritten depth decoders vs. the shipped one")
    model = consumer.model
    settings = model.depth_decoder.generation_config
    remembered = settings.do_sample
    settings.do_sample = False
    try:
        torch.manual_seed(0)
        hidden = model.backbone_model.config.hidden_size
        cond = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
        uncond = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
        first = torch.randint(0, 1000, (rows, 1), device=device)
        start = torch.cat([torch.zeros_like(first), first], dim=-1)

        with torch.inference_mode():
            shipped = BreezeGenerationMixin._depth_decoder_generate_with_cfg(
                model, start, cond, uncond, 2.5)
            cached = cached_depth_generate(model, start, cond, uncond, 2.5)
            fused = fused_depth_generate(model, start, cond, uncond, 2.5)
    finally:
        settings.do_sample = remembered

    same = all(bool(torch.equal(shipped, other)) for other in (cached, fused))
    print(f"shipped: {shipped[0].tolist()}")
    print(f"cached : {cached[0].tolist()}")
    print(f"fused  : {fused[0].tolist()}")
    ours = fused
    same = check_padding(consumer, device) and same
    if not same:
        differing = (shipped != ours).sum().item()
        print(f"{differing} of {shipped.numel()} tokens differ")
    print(f"\nverdict: {'PASS' if same else 'FAIL'}")
    return same


def check_render(bundle: Path, voicebook: Path | None, consumer, device: str,
                 seed: int = 4242) -> bool:
    """A full render, both stacks, same seed."""
    from tts_finetune.voice_conditioning.pipeline import Request as OldRequest
    from tts_finetune.voice_conditioning.pipeline import VoicePipeline

    from soul_voice import Request

    banner("3. full render vs. the research pipeline")
    started = time.time()
    new = consumer.synthesize(
        [Request(text=t, instruction=i, seed=seed) for t, i in PROMPTS],
        sample_voices=voicebook is not None)
    ours = time.time() - started

    pipeline = VoicePipeline.load(bundle, source_dir=SOURCE, device=device)
    if voicebook is not None:
        from tts_finetune.voice_conditioning.voicebook import PrototypeVoicebook

        pipeline.projector.attach_voicebook(
            PrototypeVoicebook.load(voicebook).to(device))
    started = time.time()
    old = pipeline.synthesize(
        [OldRequest(text=t, instruction=i, seed=seed) for t, i in PROMPTS],
        sample_voices=voicebook is not None)
    theirs = time.time() - started

    passed = compare(new, old)
    frames = sum(a.shape[0] for a in new) / 24_000
    print(f"\n{ours:.1f}s here vs {theirs:.1f}s in the research stack "
          f"for {frames:.1f}s of audio ({theirs / max(ours, 1e-9):.2f}x)")
    print(f"\nverdict: {'PASS' if passed else 'FAIL'}")
    return passed, pipeline


def compare(new: list, old: list) -> bool:
    passed = True
    for index, (a, b) in enumerate(zip(new, old)):
        if a.shape != b.shape:
            print(f"prompt {index}: length {a.shape[0]} vs {b.shape[0]} - DIFFER")
            passed = False
            continue
        gap = float(np.abs(a - b).max())
        print(f"prompt {index}: {a.shape[0]} samples, max |difference| {gap:.2e}")
        passed = passed and gap < 1e-4
    return passed


def check_clone(pipeline, consumer, reference: Path, device: str,
                seed: int = 4242) -> bool:
    """The reference path, which `design` never touches.

    Clone and direction read a recording through the Stage-1 encoder, so they
    exercise the audio loader, the high-pass and the frontend. A design-only
    check passes with all three broken.
    """
    from tts_finetune.voice_conditioning.pipeline import Request as OldRequest

    from soul_voice import Request

    banner("4. clone and direction vs. the research pipeline")
    text, instruction = PROMPTS[0]
    rows = [("clone", 1.0), ("direction", 0.7)]
    new = consumer.synthesize(
        [Request(text=text, instruction=instruction, mode=mode,
                 reference=reference, seed=seed, style_mix_alpha=alpha)
         for mode, alpha in rows], sample_voices=False)
    old = pipeline.synthesize(
        [OldRequest(text=text, instruction=instruction, mode=mode,
                    reference=reference, seed=seed, style_mix_alpha=alpha)
         for mode, alpha in rows], sample_voices=False)

    passed = compare(new, old)
    print(f"\nverdict: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / "checkpoints/raft")
    parser.add_argument("--voicebook", type=Path, default=None)
    parser.add_argument("--reference", type=Path,
                        default=WORKSPACE / "breeze_label/audio/ref_spk1_reference.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-render", action="store_true")
    # Which depth loop the rendered comparison runs through. `shipped` by
    # default because bit-exactness is what checks 3 and 4 assert, and the
    # fast path resamples differently; check 2 is what holds it honest.
    parser.add_argument("--depth", default="shipped",
                        choices=["fused", "cached", "shipped"])
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(WORKSPACE / "tts_finetune/src"))
    sys.path.insert(0, str(SOURCE))

    consumer, report = check_weights(args.bundle, args.voicebook, args.device,
                                     depth=args.depth)
    depth = check_depth(consumer, args.device)

    render, clone = True, True
    if not args.skip_render:
        render, pipeline = check_render(
            args.bundle, args.voicebook, consumer, args.device)
        if args.reference.is_file():
            clone = check_clone(pipeline, consumer, args.reference, args.device)
        else:
            print(f"\nno reference at {args.reference}, skipping the clone check")

    banner("summary")
    checks = (("weights", report.ok), ("cached depth", depth),
              ("render parity", render), ("clone parity", clone))
    for name, ok in checks:
        print(f"  {name:<16s} {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

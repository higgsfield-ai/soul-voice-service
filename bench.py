"""Where the time goes, per batch size.

Every row in a batch is the same request, so a run is reproducible and audio
seconds scale exactly with the batch. Real traffic has ragged lengths and the
batch runs until its longest row finishes, so treat these as an upper bound on
what batching buys.

The breakdown is measured, not apportioned: the depth decoder and the codec are
wrapped where they are called, and the backbone is what is left of generate.

    python bench.py --label before --batches 1,2,4,8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "tts_finetune/third_party/breeze-tts"
SAMPLE_RATE = 24_000
# One codec frame is 1920 samples, so 12.5 a second - not 25. A frame is the
# unit the backbone decodes once and the depth decoder fifteen times.
SAMPLES_PER_FRAME = 1920

TEXT = "We go live in thirty seconds, and I need everyone at their stations."
INSTRUCTION = "An urgent young female voice, clipped and low, pushing the pace."


class Stopwatch:
    """Wraps the two stages that can be timed where they are called."""

    def __init__(self, consumer):
        self.consumer = consumer
        self.depth = 0.0
        self.codec = 0.0
        self._patched = []

    def __enter__(self):
        model = self.consumer.model
        tokenizer = self.consumer.audio_tokenizer

        shipped_depth = model._depth_decoder_generate_with_cfg

        def timed_depth(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = shipped_depth(*args, **kwargs)
            torch.cuda.synchronize()
            self.depth += time.perf_counter() - started
            return out

        model._depth_decoder_generate_with_cfg = timed_depth
        self._patched.append((model, "_depth_decoder_generate_with_cfg", shipped_depth))

        shipped_decode = tokenizer.decode

        def timed_decode(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = shipped_decode(*args, **kwargs)
            torch.cuda.synchronize()
            self.codec += time.perf_counter() - started
            return out

        tokenizer.decode = timed_decode
        self._patched.append((tokenizer, "decode", shipped_decode))
        return self

    def __exit__(self, *exc):
        for owner, name, original in self._patched:
            setattr(owner, name, original)


def run(consumer, batch: int, seed: int = 4242) -> dict:
    from soul_voice import Request

    requests = [Request(text=TEXT, instruction=INSTRUCTION, seed=seed)
                for _ in range(batch)]
    watch = Stopwatch(consumer)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with watch:
        clips = consumer.synthesize(requests, batch_size=batch)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    seconds = sum(len(c) for c in clips) / SAMPLE_RATE
    frames = max(len(c) for c in clips) / SAMPLES_PER_FRAME
    return {
        "batch": batch,
        "wall": wall,
        "audio_seconds": seconds,
        "frames": frames,
        "rtf": seconds / wall,
        "ms_per_frame": wall / max(frames, 1) * 1000,
        "depth": watch.depth,
        "codec": watch.codec,
        "backbone": wall - watch.depth - watch.codec,
    }


def table(rows: list[dict]) -> str:
    lines = [
        f"{'batch':>5} {'clips/s':>8} {'RTF':>7} {'ms/frame':>9} "
        f"{'wall':>7} {'audio':>7} | {'backbone':>9} {'depth':>8} {'codec':>8}",
        "-" * 88,
    ]
    for row in rows:
        share = lambda part: f"{part / row['wall'] * 100:4.0f}%"  # noqa: E731
        lines.append(
            f"{row['batch']:>5} {row['batch'] / row['wall']:>8.2f} "
            f"{row['rtf']:>7.2f} {row['ms_per_frame']:>9.1f} "
            f"{row['wall']:>6.1f}s {row['audio_seconds']:>6.1f}s | "
            f"{row['backbone']:>5.1f}s {share(row['backbone']):>4} "
            f"{row['depth']:>4.1f}s {share(row['depth']):>4} "
            f"{row['codec']:>4.1f}s {share(row['codec']):>4}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "checkpoints/round1")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label", default="run")
    parser.add_argument("--depth", default="fused",
                        choices=["fused", "cached", "shipped"])
    parser.add_argument("--compile", action="store_true",
                        help="capture the depth decoder in CUDA graphs")
    parser.add_argument("--backbone-cache", action="store_true",
                        help="override the inherited config.use_cache = False")
    parser.add_argument("--out", type=Path, default=ROOT / "bench")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SOURCE))
    from soul_voice import VoiceConsumer

    consumer = VoiceConsumer.load(args.bundle, source_dir=SOURCE,
                                  device=args.device, depth=args.depth,
                                  compile=args.compile)
    if args.backbone_cache:
        consumer.model.config.use_cache = True
    batches = [int(b) for b in args.batches.split(",")]

    print("warming up", flush=True)
    run(consumer, 1)

    rows = []
    for batch in batches:
        # Capture is per shape, so each batch size pays its own compilation.
        # Warming it first is what a server would see after the first request.
        if args.compile:
            run(consumer, batch)
        row = run(consumer, batch)
        rows.append(row)
        print(f"  batch {batch:>2}: {row['wall']:.1f}s wall, "
              f"{row['rtf']:.2f}x realtime", flush=True)

    print(f"\n{args.label}\n{table(rows)}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

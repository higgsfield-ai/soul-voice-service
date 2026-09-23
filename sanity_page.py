"""Render a listening page: does this bundle still sound like itself?

Sixty clips over the three modes, laid out so the two things that can silently
go wrong are audible side by side.

`design` gets two blocks. One holds the seed still and asks for three voices in
a single call, which is the voicebook drawing three prototypes - if they sound
like one person, the voicebook did not attach. The other turns the draw off and
varies the seed, so the voice is fixed and only the reading moves.

`clone` and `direction` get three seeds each against a fixed reference, which is
played beside them. Their identity comes from the recording rather than a draw,
so it should hold across all three takes.

The work splits into independent `synthesize` calls, so it shards across GPUs
one worker each; a clip depends only on its own call, not on what else ran
beside it.

    python sanity_page.py --bundle checkpoints/round1 --gpus 8
"""

import argparse
import html
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "tts_finetune/third_party/breeze-tts"
SAMPLE_RATE = 24_000
SEEDS = (101, 202, 303)
DRAW_SEED = 4242
VOICES = 3
DIRECTION_ALPHA = 1.0

BLOCKS = ("design_draws", "design_seeds", "clone", "direction")


def slices(prompts: list[dict]) -> dict:
    return {"design_draws": prompts[:5], "design_seeds": prompts[:5],
            "direction": prompts[5:10], "clone": prompts[10:15]}


def pick_references(folder: Path, count: int = 5) -> list[Path]:
    """The longest recordings: identity is thin to take from two seconds."""
    import soundfile as sf

    everything = sorted(folder.glob("ref_spk*.wav"))
    return sorted(everything, key=lambda p: -sf.info(str(p)).duration)[:count]


def plan(prompts: list[dict], references: list[Path]) -> list[dict]:
    """Every call this page needs, as independent units of work."""
    groups = slices(prompts)
    units = []

    # One call per prompt, three rows, one seed: three prototype draws.
    for row, prompt in enumerate(groups["design_draws"]):
        units.append({
            "block": "design_draws", "sample_voices": True,
            "requests": [{"text": prompt["text"], "instruction": prompt["instruction"],
                          "seed": DRAW_SEED} for _ in range(VOICES)],
            "targets": [{"row": row, "column": voice,
                         "file": f"design_draw_{row}_{voice}.wav"}
                        for voice in range(VOICES)]})

    # One call per seed, all five prompts, draw off: fixed voice, three takes.
    for column, seed in enumerate(SEEDS):
        units.append({
            "block": "design_seeds", "sample_voices": False,
            "requests": [{"text": p["text"], "instruction": p["instruction"],
                          "seed": seed} for p in groups["design_seeds"]],
            "targets": [{"row": row, "column": column,
                         "file": f"design_seed_{row}_{seed}.wav"}
                        for row in range(len(groups["design_seeds"]))]})

    for block in ("clone", "direction"):
        group = groups[block]
        picked = [references[i % len(references)] for i in range(len(group))]
        for column, seed in enumerate(SEEDS):
            units.append({
                "block": block, "sample_voices": False,
                "requests": [{"text": p["text"], "instruction": p["instruction"],
                              "mode": block, "reference": str(reference),
                              "seed": seed, "style_mix_alpha": DIRECTION_ALPHA}
                             for p, reference in zip(group, picked)],
                "targets": [{"row": row, "column": column,
                             "file": f"{block}_{row}_{seed}.wav"}
                            for row in range(len(group))]})
    return units


def shard(units: list[dict], workers: int) -> list[list[int]]:
    """Longest unit first onto the emptiest worker, so they finish together."""
    buckets: list[list[int]] = [[] for _ in range(workers)]
    load = [0] * workers
    for index in sorted(range(len(units)), key=lambda i: -len(units[i]["requests"])):
        lightest = load.index(min(load))
        buckets[lightest].append(index)
        load[lightest] += len(units[index]["requests"])
    return buckets


def work(units: list[dict], indices: list[int], bundle: Path, voicebook: Path | None,
         device: str, out: Path) -> list[dict]:
    import soundfile as sf

    from soul_voice import Request, VoiceConsumer

    consumer = VoiceConsumer.load(bundle, source_dir=SOURCE, device=device,
                                  voicebook=voicebook)
    report = consumer.verify()
    print(report, flush=True)

    done = []
    for position, index in enumerate(indices):
        unit = units[index]
        started = time.time()
        clips = consumer.synthesize(
            [Request(**row) for row in unit["requests"]],
            sample_voices=unit["sample_voices"])
        for target, clip in zip(unit["targets"], clips):
            sf.write(str(out / target["file"]), clip, SAMPLE_RATE)
            done.append({"block": unit["block"], **target})
        print(f"  [{device}] {position + 1}/{len(indices)} {unit['block']} "
              f"{len(clips)} clips in {time.time() - started:.0f}s", flush=True)
    return done


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy import signal

        mono = signal.resample_poly(mono, SAMPLE_RATE, rate)
    return np.asarray(mono, dtype=np.float32)


STYLE = """
* { box-sizing: border-box; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 1180px; padding: 40px 28px 80px; color: #1b1b1f;
       background: #fbfbfd; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.02em; }
h2 { font-size: 19px; margin: 44px 0 4px; letter-spacing: -0.01em; }
.lede, .note { color: #5c5f6b; margin: 0 0 18px; max-width: 760px; }
.note { font-size: 14px; }
.meta { display: flex; flex-wrap: wrap; gap: 10px 26px; padding: 14px 18px; margin: 18px 0 8px;
        background: #fff; border: 1px solid #e3e4ea; border-radius: 10px; font-size: 13px; }
.meta b { font-weight: 600; }
.meta .bad { color: #b3261e; }
table { width: 100%; border-collapse: collapse; margin-top: 14px; background: #fff;
        border: 1px solid #e3e4ea; border-radius: 10px; overflow: hidden; }
th { text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
     color: #6b6e7b; font-weight: 600; padding: 11px 14px; background: #f4f4f7;
     border-bottom: 1px solid #e3e4ea; }
td { padding: 13px 14px; border-bottom: 1px solid #eeeff3; vertical-align: top; }
tr:last-child td { border-bottom: none; }
.text { font-weight: 550; margin-bottom: 5px; }
.instruction { color: #5c5f6b; font-size: 13px; }
.reference { color: #6b6e7b; font-size: 12px; margin-top: 7px; }
audio { width: 218px; height: 34px; display: block; }
td.clip { width: 236px; }
"""

SECTIONS = (
    ("design_draws", "Design - three voices for one description",
     f"One call, three rows, seed {DRAW_SEED} throughout. Each row draws its own "
     "prototype from the voicebook, so the three should be different people reading "
     "the same line. If they sound like one person, the voicebook is not attached.",
     tuple(f"voice {i + 1}" for i in range(VOICES))),
    ("design_seeds", "Design - one voice, three readings",
     "The draw is off, so the projector returns its averaged voice and identity is "
     "fixed. Only the seed moves, so these should be the same speaker giving three "
     "different takes.",
     tuple(f"seed {s}" for s in SEEDS)),
    ("clone", "Clone - identity from a recording",
     "Identity and delivery both come from the reference, which is the first player "
     "on each row. The three takes differ only by seed, and the speaker should hold "
     "across them and match the reference.",
     tuple(f"seed {s}" for s in SEEDS)),
    ("direction", "Direction - that speaker, delivered to order",
     f"Identity still comes from the reference, but delivery follows the description "
     f"at style_mix_alpha {DIRECTION_ALPHA}. Against the clone block above: same kind "
     "of source, different delivery.",
     tuple(f"seed {s}" for s in SEEDS)),
)


def page(blocks: dict, header: dict) -> str:
    def escape(value):
        return html.escape(str(value))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>soul-voice sanity check</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>soul-voice sanity check</h1>",
        f"<p class='lede'>{escape(header['count'])} clips from "
        f"<code>{escape(header['bundle'])}</code>, rendered through this package. "
        "Every block varies exactly one thing.</p>",
        "<div class='meta'>",
    ]
    for label, value in header["facts"]:
        bad = " class='bad'" if "absent" in str(value) else ""
        parts.append(f"<span><b>{escape(label)}</b> <span{bad}>{escape(value)}</span></span>")
    parts.append("</div>")

    for key, title, note, columns in SECTIONS:
        rows = blocks[key]
        parts.append(f"<h2>{escape(title)}</h2><p class='note'>{escape(note)}</p><table>")
        headers = "".join(f"<th>{escape(c)}</th>" for c in columns)
        reference_header = "<th>reference</th>" if rows[0].get("reference") else ""
        parts.append(f"<tr><th>prompt</th>{reference_header}{headers}</tr>")
        for row in rows:
            cells = [
                "<td><div class='text'>" + escape(row["text"]) + "</div>"
                "<div class='instruction'>" + escape(row["instruction"]) + "</div></td>"]
            if row.get("reference"):
                cells.append(
                    f"<td class='clip'><audio controls preload='none' src='{row['reference']}'>"
                    f"</audio><div class='reference'>{escape(row['reference_name'])}</div></td>")
            cells.extend(
                f"<td class='clip'><audio controls preload='none' src='{clip}'></audio></td>"
                for clip in row["clips"])
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


def assemble(prompts: list[dict], references: list[Path], rendered: list[dict],
             header: dict, out: Path) -> int:
    import soundfile as sf

    groups = slices(prompts)
    blocks: dict = {}
    for key, _, _, columns in SECTIONS:
        group = groups[key]
        picked = [references[i % len(references)] for i in range(len(group))]
        rows = []
        for index, prompt in enumerate(group):
            clips = [None] * len(columns)
            for item in rendered:
                if item["block"] == key and item["row"] == index:
                    clips[item["column"]] = item["file"]
            row = {"text": prompt["text"], "instruction": prompt["instruction"],
                   "clips": [c for c in clips if c]}
            if key in ("clone", "direction"):
                name = f"{key}_reference_{index}.wav"
                sf.write(str(out / name), read_wav(picked[index]), SAMPLE_RATE)
                row["reference"] = name
                row["reference_name"] = picked[index].name
            rows.append(row)
        blocks[key] = rows

    count = sum(len(row["clips"]) for rows in blocks.values() for row in rows)
    header["count"] = count
    (out / "index.html").write_text(page(blocks, header))
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "checkpoints/round1")
    parser.add_argument("--voicebook", type=Path, default=WORKSPACE / "voicebook_tuned.pt")
    parser.add_argument("--prompts", type=Path, default=Path("/tmp/sanity_prompts.json"))
    parser.add_argument("--references", type=Path, default=WORKSPACE / "breeze_label/audio")
    parser.add_argument("--out", type=Path, default=ROOT / "sanity")
    parser.add_argument("--gpus", type=int, default=0, help="0 means every visible GPU")
    parser.add_argument("--shard", type=int, default=None, help="internal: worker index")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SOURCE))

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text())
    if len(prompts) < 15:
        raise SystemExit(f"need 15 prompts, {args.prompts} has {len(prompts)}")
    references = pick_references(args.references)
    units = plan(prompts, references)
    book = args.voicebook if args.voicebook.is_file() else None

    import torch

    workers = args.gpus or torch.cuda.device_count()

    if args.shard is not None:
        done = work(units, shard(units, workers)[args.shard], args.bundle, book,
                    args.device, args.out)
        (args.out / f"_shard_{args.shard}.json").write_text(json.dumps(done))
        return 0

    for stale in args.out.glob("_shard_*.json"):
        stale.unlink()
    buckets = shard(units, workers)
    print(f"{len(units)} calls, {sum(len(u['requests']) for u in units)} clips, "
          f"{workers} GPUs ({[sum(len(units[i]['requests']) for i in b) for b in buckets]} "
          f"clips each)\n")

    started = time.time()
    running = []
    for index in range(workers):
        log = (args.out / f"_shard_{index}.log").open("w")
        running.append((index, subprocess.Popen(
            [sys.executable, __file__, "--bundle", str(args.bundle),
             "--voicebook", str(args.voicebook), "--prompts", str(args.prompts),
             "--references", str(args.references), "--out", str(args.out),
             "--gpus", str(workers), "--shard", str(index),
             "--device", f"cuda:{index}"], stdout=log, stderr=subprocess.STDOUT), log))

    failed = []
    for index, process, log in running:
        if process.wait() != 0:
            failed.append(index)
        log.close()
    if failed:
        for index in failed:
            print(f"--- shard {index} failed, tail of its log ---")
            print("\n".join((args.out / f"_shard_{index}.log").read_text().splitlines()[-12:]))
        return 1

    rendered = [item for index in range(workers)
                for item in json.loads((args.out / f"_shard_{index}.json").read_text())]
    elapsed = time.time() - started

    # Read back from a worker's log rather than loading the model again here.
    summary = (args.out / "_shard_0.log").read_text()
    prototypes = next((line.split()[1] for line in summary.splitlines()
                       if line.startswith("voicebook")), "?")
    header = {
        "bundle": args.bundle,
        "facts": [
            ("voicebook", "absent" if prototypes == "absent" else f"{prototypes} prototypes"),
            ("guidance", 2.5), ("temperature", 0.9),
            ("GPUs", workers),
            ("rendered", f"{elapsed / 60:.1f} min wall clock"),
        ],
    }
    count = assemble(prompts, references, rendered, header, args.out)
    print(f"\n{count} clips in {elapsed / 60:.1f} min -> {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

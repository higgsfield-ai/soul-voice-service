"""Render all modes on one loaded GPU model and compare wrapper vs direct inference.

Run from the repository root:
    python scripts/gpu_smoke.py --reference speaker.wav --output outputs/gpu-smoke
This needs no AWS credentials and does not send queue messages.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_service.engine import VoiceEngine
from voice_service.schema import Payload
from voice_service.settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu-smoke"))
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error("--reference must be an existing audio recording")
    args.output.mkdir(parents=True, exist_ok=True)
    engine = VoiceEngine(Settings.from_env())
    report = {"gpu": engine.gpu_type, "modes": {}}
    examples = Path(__file__).resolve().parents[1] / "examples"
    for mode in ("design", "clone", "direction"):
        payload = Payload.model_validate_json((examples / f"{mode}.json").read_text())
        reference = None if mode == "design" else args.reference
        output = args.output / f"{mode}.wav"
        metadata = engine.render(payload, reference, output)
        saved, rate = sf.read(output, dtype="float32")
        previous_sampling = engine.consumer.sampling
        engine.consumer.sampling = replace(previous_sampling, **metadata["sampling"])
        try:
            engine.consumer._apply_sampling()
            request = engine.request_type(
                **payload.voice_config.model_dump(exclude=set(metadata["sampling"])), reference=reference
            )
            (direct,) = engine.consumer.synthesize([request], batch_size=1)
        finally:
            engine.consumer.sampling = previous_sampling
            engine.consumer._apply_sampling()
        np.testing.assert_array_equal(saved, direct)
        assert rate == 24000 and np.isfinite(saved).all() and np.any(saved != 0)
        Path(str(output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
        report["modes"][mode] = {
            "matches_direct_inference": True,
            "samples": len(saved),
            "sample_rate": rate,
            "duration_seconds": len(saved) / rate,
        }
        print(f"{mode}: {len(saved) / rate:.2f}s audio; matches direct inference", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

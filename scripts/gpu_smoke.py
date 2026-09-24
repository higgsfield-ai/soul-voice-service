"""Render all checkpoints and modes in one worker and compare wrapper vs direct inference.

Run from the repository root:
    python scripts/gpu_smoke.py --reference speaker.wav --output outputs/gpu-smoke
This needs no AWS credentials and does not send queue messages.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import get_args

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.pipeline import Pipeline
from src.schemas import Payload
from src.schemas.voice import Checkpoint
from src.settings import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/gpu-smoke'))
    parser.add_argument(
        '--checkpoints', nargs='+', choices=get_args(Checkpoint), default=list(get_args(Checkpoint))
    )
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error('--reference must be an existing audio recording')
    args.output.mkdir(parents=True, exist_ok=True)
    engine = Pipeline(settings)
    report = {'gpu': engine.gpu_type, 'checkpoints': {}}
    examples = Path(__file__).resolve().parents[1] / 'examples'
    for checkpoint in args.checkpoints:
        checkpoint_output = args.output / checkpoint
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        report['checkpoints'][checkpoint] = {}
        for mode in ('design', 'clone', 'direction'):
            payload = Payload.model_validate_json((examples / f'{mode}.json').read_text())
            payload.voice_config.checkpoint = checkpoint
            reference = None if mode == 'design' else args.reference
            output = checkpoint_output / f'{mode}.wav'
            metadata = engine(reference, output, config=(payload).voice_config)
            saved, rate = sf.read(output, dtype='float32')
            consumer = engine.get_consumer(checkpoint)
            previous_sampling = consumer.sampling
            consumer.sampling = replace(previous_sampling, **metadata['sampling'])
            try:
                consumer._apply_sampling()
                request = engine.request_type(
                    **payload.voice_config.model_dump(exclude=set(metadata['sampling']) | {'checkpoint'}),
                    reference=reference,
                )
                (direct,) = consumer.synthesize([request], batch_size=1)
            finally:
                consumer.sampling = previous_sampling
                consumer._apply_sampling()
            np.testing.assert_array_equal(saved, direct)
            assert rate == 24000 and np.isfinite(saved).all() and np.any(saved != 0)
            Path(str(output) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
            report['checkpoints'][checkpoint][mode] = {
                'matches_direct_inference': True,
                'samples': len(saved),
                'sample_rate': rate,
                'duration_seconds': len(saved) / rate,
            }
            print(
                f'{checkpoint}/{mode}: {len(saved) / rate:.2f}s audio; matches direct inference', flush=True
            )
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

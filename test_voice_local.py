"""Render a voice payload locally without SQS or S3."""

import argparse
import json
from pathlib import Path

from src.core.pipeline import Pipeline
from src.schemas import Payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = Payload.model_validate_json(args.payload.read_text())
    if payload.voice_config.mode != 'design' and args.reference is None:
        parser.error('clone and direction require --reference')
    if args.reference is not None and not args.reference.is_file():
        parser.error('--reference must name an existing audio file')
    if args.reference is not None and args.output.resolve() == args.reference.resolve():
        parser.error('--output must not overwrite --reference')
    pipeline = Pipeline()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference = args.reference if payload.voice_config.mode != 'design' else None
    metadata = pipeline(reference, args.output, config=payload.voice_config)
    metadata['job_id'] = payload.job_id
    Path(str(args.output) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

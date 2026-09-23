"""python -m voice_service {worker,validate,render}."""

import argparse
import json
import logging
import signal
from pathlib import Path

from .schema import Payload
from .settings import Settings


def main():
    parser = argparse.ArgumentParser(description="Soul Voice SQS worker and local inference")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("worker", help="load models, then consume SQS jobs")
    validate = commands.add_parser("validate", help="validate without loading models")
    validate.add_argument("payload", type=Path)
    render = commands.add_parser("render", help="render locally, without AWS")
    render.add_argument("payload", type=Path)
    render.add_argument("--reference", type=Path)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command in {"validate", "render"}:
        payload = Payload.model_validate_json(args.payload.read_text())
        if args.command == "validate":
            print(payload.model_dump_json(indent=2))
            return
        if payload.voice_config.mode != "design" and args.reference is None:
            parser.error("clone and direction require --reference for local rendering")
        if args.reference is not None and not args.reference.is_file():
            parser.error("--reference must name an existing audio file")
        if args.reference is not None and args.output.resolve() == args.reference.resolve():
            parser.error("--output must not overwrite --reference")
    settings = Settings.from_env()
    if args.command == "worker" and not settings.queue_url:
        parser.error("SQS_QUEUE_URL is required")
    from .engine import VoiceEngine

    engine = VoiceEngine(settings)
    if args.command == "render":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        reference = args.reference if payload.voice_config.mode != "design" else None
        metadata = engine.render(payload, reference, args.output)
        metadata["job_id"] = payload.job_id
        Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(args.output)
        return
    from .worker import Worker

    worker = Worker(settings, engine)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: worker.stop.set())
    worker.run()


if __name__ == "__main__":
    main()

# Verification status

Updated on 2026-09-25 for the `simplify-service` branch.

Local checks:

- 82 tests passed. They cover all three checkpoints and all three modes, default
  selection, model reuse, sampling overrides/restoration, reference routing,
  float32 WAV samples, result metadata and the emulated SQS/S3 round trip.
- Delivery tests cover upload/result/ack order, cloud retries, failed inference,
  visibility renewal failures, FIFO results and shutdown during polling.
- The folder downloader is tested for all four model directories, pagination,
  exact prefix matching, cached hashes, corrupt/interrupted downloads, missing
  directories and files uploaded without SHA-256 metadata.
- Entry-point tests cover the worker, payload validation/submission and local
  rendering. The original consumer's sampling hook is exercised on CPU when
  inference dependencies are installed; that check is skipped in the lightweight
  test environment. Neural-network synthesis is mocked in the local suite.
- Ruff lint/format checks passed. `uv lock --check --offline` passed without
  changing dependency versions. The Python wheel built offline; its new entry
  points imported successfully without loading GPU models. Payload dry runs and
  local-render/GPU-helper help commands passed.
- Both compose files parse with the new service/download commands. Docker is not
  installed on this laptop, so the image build remains a GPU-host check.
- The supplied `soul_voice/` and vendored Breeze source are unchanged.

The refactor uses the Recast `src/` layout and separate download compose file.
It replaces the inventory/publish/verify CLI with folder downloads from the
existing model release. The request's `voice_config`, sampling options, checkpoint
selection and result envelope are preserved. General text-length, extra-field and
S3-location-collision validation was removed; voice-specific requirements remain.

Earlier checks and provisioning:

- Loaded the supplied text tokenizer locally: 262,158 tokens. Constructed Breeze
  on the meta device and checked both supplied shards: 1,116 keys, with no missing
  or unexpected keys and no shape mismatches.
- Resolved `local-soul-voice` and `local-soul-voice-result` and read their attributes
  in AWS account `582881730701`, region `eu-north-1`.
- Created private bucket `soul-voice-service` with public access blocked, ACLs
  disabled and AES256 encryption. Published 38 files (9,976,678,159 bytes) under
  `models/soul-voice/eb13a103470bdb26/`. The original publication verified remote
  sizes and SHA-256 metadata and downloaded/hashed 23 JSON files (34,686,187 bytes).
  The old release inventory remains in S3; the folder downloader does not use it.

Still pending on the new Nebius machine:

- Linux Docker image build, full model download with the refactored downloader,
  and NVIDIA GPU loading/inference.
- Listening to fresh design, clone and direction renders.
- GPU wrapper-vs-direct comparisons across checkpoints using `scripts/gpu_smoke.py`.
- Real SQS/S3 voice jobs using the dedicated voice queues.

No cloud jobs or model uploads were performed during this refactor. Original
training-pipeline parity claims have not been reproduced here.

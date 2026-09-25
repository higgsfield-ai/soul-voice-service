# Verification status

Updated on 2026-09-25 for the `simplify-service` branch. GPU and live AWS tests
used source commit `43dddc4`; no service-code changes were needed during testing.

## GPU and deployment

- Built the repository's Linux Docker image from scratch on Nebius. Verified
  PyTorch 2.9.1+cu130, Transformers 4.57.3, driver 580.173.02 and an NVIDIA L40S
  exposing 46,068 MiB of GPU memory.
- Downloaded all 38 model files (9,976,678,159 bytes) using
  `docker-compose.download.yml` and the real `src.utils.download` entry point.
  File sizes and SHA-256 metadata passed verification. A second run verified and
  reused the cached files without downloading them again.
- `scripts/gpu_smoke.py` passed for all nine combinations of `base`, `raft`,
  `round1` and `design`, `clone`, `direction`. Saved float32 WAV samples exactly
  matched direct calls to the supplied inference API with the same settings.
- All three checkpoint stacks remained loaded in one process. Observed GPU
  usage with all three loaded was about 27.3 GiB; there were no out-of-memory
  failures. All bundles passed the original consumer's strict model checks.
- Additional `round1` checks passed for direction blending at alpha 0 and 1,
  all six sampling overrides, restoration of defaults, and repeated requests.
- Representative design requests passed direct-inference comparisons and
  repeated-sample equality with `fused`, `cached`, and `shipped` decoders, and
  with `VOICE_COMPILE=true` and `TORCHINDUCTOR_COMPILE_THREADS=2`. Compilation
  completed without falling back to eager.
  Different decoder settings can produce different sampled audio, as documented
  by the supplied implementation; equality was checked within each setting.

## Live SQS and S3

Used the real `local-soul-voice` and `local-soul-voice-result` queues in
`eu-north-1`, with media under unique `tests/` prefixes in `soul-voice-service`.
The worker ran through the normal Docker Compose entry point.

- A 15-job suite passed: all nine checkpoint/mode combinations, omitted
  checkpoint defaulting to `round1`, per-job sampling overrides, invalid payload,
  missing S3 reference, malformed audio, and a successful recovery request.
  Twelve jobs completed and three returned the expected terminal failures.
- The worker loaded the checkpoints in a different order from the local GPU
  helper. All nine resulting SQS/S3 WAVs still matched the local outputs exactly.
- Verified job IDs and status messages, model selection, effective sampling,
  audio format and samples, S3 result URIs, and equality between the uploaded
  metadata JSON and `meta.pipeline` in the completion message.
- Overrides and failed inference did not affect the subsequent repeated request;
  it reproduced the earlier samples exactly. All jobs reported zero retries.
- A further job completed during graceful shutdown: SIGTERM was sent after
  `in_progress`, and the worker finished inference, uploaded its outputs,
  published completion, acknowledged the request, and exited with status 0.
- The test worker used a 60-second visibility timeout, exercising its renewal
  thread during longer jobs. Both queues were confirmed empty after their
  approximate statistics settled. The worker was left stopped after testing.

The main live run's media prefix is
`s3://soul-voice-service/tests/simplify-20260925T131027Z-1ef5d4/`.
The shutdown job used `tests/simplify-shutdown-1790342205/` in the same bucket.

## Audio examples and limits

Audio, payloads, metadata, reports, logs and a listening page were recovered to
`voice-test-outputs/20260925-l40s/` in the local test workspace. The WAVs and test
credentials are not committed to the service repository.

- An independent Whisper speech-recognition check recovered the requested words
  from all 22 checked clips, including the nine main examples, after normalizing
  punctuation, case, and `thirty`/`30`. This checks intelligibility; human
  listening is still needed to assess speaker identity, delivery and perceptual
  quality.
- Clone and direction used the supplied synthesized `clip_01.wav` as reference.
- Regenerated the original example's text, instruction, checkpoint and seed
  76871. The new 10.72-second render matches direct inference on this machine,
  but differs from the supplied 8.48-second PCM16 recording. The original clip
  does not record all runtime and decoder settings, so it is a listening
  reference rather than an established numerical ground truth.
- Comparisons use the supplied inference package. Parity with the unavailable
  original training pipeline has not been reproduced. Voicebook casting remains
  untested because no voicebook weights were supplied.
- These checks cover short utterances and sequential jobs. Sustained production
  load and live AWS transport-failure injection were not tested; cloud retry and
  visibility-loss behavior are covered by the local tests below.

## Local checks

- 82 tests passed, including a fresh run before GPU testing. They cover all three
  checkpoints and modes, default selection, model reuse, sampling restoration,
  reference routing, float32 WAV samples, metadata and emulated SQS/S3 delivery.
- Delivery tests cover upload/result/ack order, cloud retries, failed inference,
  visibility renewal failures, FIFO results and shutdown during polling.
- Downloader tests cover all four model directories, pagination, exact prefix
  matching, cached hashes, corrupt/interrupted downloads, missing directories
  and files uploaded without SHA-256 metadata.
- Entry-point tests cover the worker, payload submission and local rendering.
  Neural synthesis is mocked in this suite; the original sampling hook was
  exercised on CPU with inference dependencies installed. Real neural synthesis
  is covered by the GPU checks above.
- Ruff lint/format checks and `uv lock --check --offline` passed. The wheel built
  offline and its new entry points imported successfully. Payload dry runs and
  local-render/GPU-helper help commands passed.
- The supplied `soul_voice/`, vendored Breeze implementation, dependency lock,
  payload contract and model weights remain unchanged by the simplification.

## Earlier model checks

- Loaded the text tokenizer locally: 262,158 tokens. Constructed Breeze on the
  meta device and checked both backbone shards: 1,116 keys, with no missing or
  unexpected keys and no shape mismatches.
- Created the private `soul-voice-service` bucket with public access blocked,
  ACLs disabled and AES256 encryption. Models were published under
  `models/soul-voice/eb13a103470bdb26/`. The old publication inventory remains in
  S3; the simplified downloader syncs the four model folders without using it.

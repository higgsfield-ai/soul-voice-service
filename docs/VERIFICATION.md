# Verification status

Verification updated on 2026-09-25:

- 85 tests passed, primarily using emulated SQS/S3 and a synthesis double: all three modes,
  references, metadata/result consistency, status publication before acknowledgement,
  failure and retry paths, visibility lease failure, FIFO result publication,
  float WAV preservation and checksummed model publishing/downloading.
- All three checkpoint choices work through payload validation and the emulated
  SQS/S3 round trip for each of the three modes. Adapter tests alternate checkpoint
  requests in one engine, verify consumer reuse and metadata, preserve sampling
  defaults, and retry failed loads without replacing successfully cached models.
- Sampling controls are checked for per-job overrides, default preservation,
  zero top-k cutoffs, effective metadata values, and restoration after generation
  succeeds or fails. All three decoder modes (`fused`, `cached`, `shipped`) and
  compilation flags are checked at the original loader boundary.
- With the inference dependencies installed, a CPU check exercises the original
  `VoiceConsumer._apply_sampling()` against model configuration objects and confirms
  that both backbone and depth settings are applied and restored. This does not
  run the neural network or establish GPU/audio correctness. That check is skipped
  when running with only the lightweight test dependencies.
- Ruff lint and formatting checks passed for the new service, tests and GPU helper.
- The supplied `soul_voice` source files are unchanged.
- The model inventory contains 38 files and 9,976,678,159 bytes, including
  all three checkpoint bundles and shared backbone/tokenizers/codec. SHA-256 values
  were generated from the actual supplied files.
- Installed the locked inference dependencies on macOS and imported Soul Voice,
  the vendored Breeze model and Qwen codec successfully. The upstream codec emits
  a missing-SoX warning on this laptop; the Dockerfile installs SoX.
- Constructed a Breeze model on the meta device from the supplied config and
  compared its state shapes to both supplied backbone safetensor shards:
  **1,116 keys, no missing/unexpected keys, no shape mismatches**.
- Loaded the supplied text tokenizer locally: 262,158 tokens.

Still pending:

- Linux Docker image build and NVIDIA GPU loading/inference with the new service.
- Listening to fresh design, clone and direction renders.
- `scripts/gpu_smoke.py` wrapper-vs-direct comparisons on the GPU.
- Real SQS/S3 end-to-end jobs using the new voice queues and machine.
- Complete model download on the new GPU host (the real S3 download check so far
  covers configuration/tokenizer JSON files).

The development queues `local-soul-voice` and `local-soul-voice-result` were resolved
and their attributes read successfully in account `582881730701`, region `eu-north-1`.
The private model bucket `soul-voice-service` was created in that same account,
with public access blocked, ACLs disabled and AES256 encryption enabled. All 38
model files (9,976,678,159 bytes) were published using the service publisher to
`models/soul-voice/eb13a103470bdb26/`. The remote inventory matches `models.json`;
every object's size, SHA-256 metadata and encryption setting were checked.
The actual service downloader fetched all 23 configuration/tokenizer JSON files
(34,686,187 bytes) into a fresh local directory and verified their SHA-256 hashes.
The complete remote weight files have not been downloaded back to this laptop.

No voice job has been sent to a real queue yet. The user will provide a new SSH
machine for GPU and live job verification. The original documentation's training
parity claims belong to the supplied model handoff; they have not been reproduced here.

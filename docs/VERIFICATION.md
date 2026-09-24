# Verification status

Local checks completed on 2026-09-23:

- 67 tests passed, primarily using emulated SQS/S3 and a synthesis double: all three modes,
  references, metadata/result consistency, status publication before acknowledgement,
  failure and retry paths, visibility lease failure, FIFO result publication,
  float WAV preservation and checksummed model publishing/downloading.
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
- The default model inventory contains 24 files and 8,537,540,386 bytes, including
  the round1 conditioners and shared backbone/tokenizers/codec. SHA-256 values
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
- Initial R2 model publication/download against the chosen real bucket.

The user will provide a new SSH machine and SQS queues. No voice jobs were sent to
the old queues and no weights were uploaded during this implementation. The
original documentation's training parity claims belong to the supplied model
handoff; they have not been reproduced here.

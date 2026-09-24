# Soul Voice service

A separate GPU service for voice design, cloning and direction. It wraps the
supplied `soul_voice` inference package with the same transport pattern as the
mask service: **SQS request → S3 reference audio → inference → S3 WAV + JSON →
SQS result**. Design jobs need no input audio.

The original model documentation is preserved in [docs/INFERENCE.md](docs/INFERENCE.md).
The inference implementation is unchanged. This service does not run ComfyUI or
Breeze's HTTP server, and it has no dependency on the mask service repository.

## Request contract

Send a JSON message to the worker's `SQS_QUEUE_URL`. The backend consumes the
queue specified by `result_queue_url`. These are separate queues.

```json
{
  "job_id": "voice-design-001",
  "task_type": "voice",
  "s3_region": "eu-north-1",
  "sqs_region": "eu-north-1",
  "result_queue_url": "https://sqs.eu-north-1.amazonaws.com/ACCOUNT_ID/voice-results",
  "dst_bucket_audio_pair": ["YOUR_MEDIA_BUCKET", "voice-tests/design/audio.wav"],
  "voice_config": {
    "text": "We go live in thirty seconds.",
    "instruction": "A warm, calm female voice, speaking clearly and unhurriedly.",
    "mode": "design",
    "seed": 4242,
    "style_mix_alpha": 1.0
  }
}
```

| Field | Meaning |
| --- | --- |
| `voice_config.text` | Words to speak; required, up to 16,000 characters. |
| `voice_config.instruction` | Description of the voice and delivery; required in every mode, up to 4,000 characters. |
| `voice_config.mode` | `design` (default), `clone`, or `direction`. |
| `voice_config.seed` | Random seed, 0 through 2³²−1; default 0. |
| `voice_config.style_mix_alpha` | For direction: 0 uses reference delivery, 1 uses instructed delivery, intermediate values blend. Default 1; ignored by clone/design. |
| `voice_config.temperature` | Sampling temperature for the backbone; positive, default 0.9. |
| `voice_config.top_k` | Number of candidate tokens retained for backbone sampling; default 50. Use 0 to disable the top-k cutoff. |
| `voice_config.depth_temperature` | Sampling temperature for the depth decoder; positive, default 0.9. |
| `voice_config.depth_top_k` | Candidate-token cutoff for the depth decoder; default 50. Use 0 to disable it. |
| `voice_config.guidance_scale` | Strength of guidance from the instructed prompt; default 2.5. A value of 1 disables the guidance branch. |
| `voice_config.max_new_tokens` | Positive generation-token limit for this job. Defaults to `VOICE_MAX_NEW_TOKENS` (1024 unless configured). |
| `src_bucket_audio_pair` | `[bucket, key]` of reference audio; required for clone/direction, unused by design. Use WAV or FLAC readable by libsndfile. |
| `dst_bucket_audio_pair` | `[bucket, key]` for the generated WAV; required. |
| `dst_bucket_metadata_pair` | Optional `[bucket, key]` for metadata. Default: audio key plus `.json`. |
| `s3_region` | Region of job-media storage; there is no fixed media bucket. |
| `sqs_region` | Region of the result queue; the input queue region is worker configuration. |

For `clone`, identity and delivery come from the reference. For `direction`,
identity comes from the reference while delivery is blended with the instruction.
The original inference code converts reference audio to mono 24 kHz and uses a
seeded crop of up to eight seconds. The service does not pre-crop it differently.

One request produces one utterance. All six sampling settings are optional fields
inside `voice_config`; omitted or null values inherit the supplied consumer's
defaults. `VOICE_MAX_NEW_TOKENS` sets the worker's default generation limit, and
`voice_config.max_new_tokens` takes precedence for that job. The environment value
is a default, not a hard upper bound. Long text can reach the generation limit;
there is no automatic sentence splitting.

For example, add `"temperature": 0.8`, `"top_k": 40`, or `"guidance_scale": 2.0`
alongside `text` and `mode` in `voice_config`. Settings apply to that job only,
including after a failed generation. Metadata's `sampling` object records the
effective values used, including defaults; `voice_config` retains the request's
options. The original inference package applies these settings.

See ready-to-edit [design](examples/design.json), [clone](examples/clone.json)
and [direction](examples/direction.json) payloads. The full machine-readable
schema is [docs/payload.schema.json](docs/payload.schema.json). Replace placeholders
and use a new job ID and output keys for each job.

```bash
python -m voice_service validate examples/design.json
aws sqs send-message --region eu-north-1 \
  --queue-url "$SQS_QUEUE_URL" --message-body file://examples/design.json
```

This is the new voice service's contract. It follows the previous result envelope;
the backend still needs to route voice requests and consume the `audio` result.
No backend implementation is included here.

## Results and metadata

The worker sends `{"fnf_job_id":"...","status":"in_progress"}` when starting,
then either `completed` or `failed` to the request's result queue.

A completion message has this shape (abbreviated):

```json
{
  "fnf_job_id": "voice-design-001",
  "status": "completed",
  "result_urls": {
    "audio": "s3://YOUR_MEDIA_BUCKET/voice-tests/design/audio.wav",
    "metadata": "s3://YOUR_MEDIA_BUCKET/voice-tests/design/audio.wav.json"
  },
  "scores": null,
  "bboxes": null,
  "meta": {
    "gpu_provider": "nebius",
    "gpu_type": "NVIDIA H100 80GB HBM3",
    "gpu_count": 1,
    "queue_url": "https://sqs.eu-north-1.amazonaws.com/ACCOUNT_ID/voice-requests",
    "retry_count": 0,
    "instance_name": "voice-worker-1",
    "real_inference_time": 12.3,
    "pipeline": {}
  }
}
```

`meta.pipeline` contains the **same metadata object** uploaded to S3, including
`schema_version`, `job_id`, `task_type`, `voice_config`, `model_version`, decoding
settings, reference S3 URI (or null), sample rate, channels, sample count, duration,
peak amplitude and inference time. Config values occur together in `voice_config`.
The output is a mono, 24 kHz, float32 WAV, preserving model samples without PCM16
rounding or clipping. Result URLs are S3 URIs, not public or presigned download URLs.

A failure message contains `fnf_job_id`, `status: "failed"`, `fail_reason`, and
worker `meta`. It does not claim output URLs are complete.

## Models and Breeze

The default bundle is the supplied **round1** checkpoint. `models.json` inventories
its files and the shared Breeze backbone, text tokenizer and Qwen audio codec,
including sizes and SHA-256 checksums. The older `base` and `raft` bundles remain
available locally but are not needed by this default service. Model weights and
credentials are excluded from Git and Docker build contexts.

A bundle is a saved training version of the voice-conditioning components, not a
different task. All three supplied bundles can serve design, clone and direction:

| Bundle | Training version |
| --- | --- |
| `base` | Stage-3 v5 phase-3 checkpoint, step 2000. |
| `raft` | RAFT round-0 refinement of `base`, step 128. |
| `round1` | GRPO round-1 refinement of `raft`, step 120; the service default. |

They share `shared/backbone` and `shared/base_checkpoint` (tokenizer and codec).
Set `VOICE_BUNDLE=checkpoints/raft`, for example, to select another supplied version
when starting the worker. This requires that bundle's files to be provisioned too:
changing the environment variable alone does not change `models.json` or download
additional weights. The current inventory provisions `round1` and its shared
dependencies. Bundle selection applies to the worker, not individual jobs.

The public Breeze inference source is vendored, unmodified, at a fixed revision
under `third_party/breeze-tts`; see [third_party/README.md](third_party/README.md).
Weights are loaded from local paths. The container sets Hugging Face offline mode,
so it does not silently replace the supplied fine-tuned package with an HF release.

No voicebook was supplied. The service uses the consumer's default averaged design
latents (`sample_voices=false`); optional prototype-based voice casting is not
exposed by this request contract.

Publish the supplied models **once**, from this checkout with its `checkpoints`
folder and R2 credentials. This is an explicit operation, not part of image builds:

```bash
cp .env.example .env
# Fill in R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.
uv sync --locked --only-group test
uv run --no-sync python -m voice_service.models verify
uv run --no-sync python -m voice_service.models publish
```

The publisher uses the release prefix recorded in `models.json`, checks all local
files first, skips matching remote objects, and refuses to replace conflicting
objects. It includes Breeze and the codec; do not upload another HF copy separately.
No model objects have been published merely by adding this service to the repo.

On a new machine the downloader recreates `round1/` and `shared/` beneath
`checkpoints/`, preserving the bundle's relative paths. It verifies SHA-256,
reuses valid files and replaces a downloaded file atomically only after verification.

```bash
python -m voice_service.models download
python -m voice_service.models verify
```

If intentionally changing model weights, regenerate and review the inventory
before publishing the new release:

```bash
python -m voice_service.models inventory --root checkpoints --bundle round1
```

R2 is for **model distribution**. Job input/output audio uses **Amazon S3** through
the normal AWS credential chain. R2 credentials never replace the worker's AWS
credentials. Preserve the supplied third-party model licenses; see the upstream
[Breeze weight terms](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).

## Run on a GPU machine

Use an NVIDIA CUDA GPU, Docker with NVIDIA Container Toolkit and a driver supporting
CUDA 13 (R580 or newer on Linux). The locked Linux runtime uses PyTorch 2.9.1+cu130,
Transformers 4.57.3 and qwen-tts 0.1.1. The lockfile targets Linux x86-64 and macOS
Apple Silicon; the laptop path is for development/tests, not GPU synthesis.

1. Copy/clone this repository to the GPU machine.
2. Create `.env` from `.env.example`. Set AWS credentials (including session token
   for temporary credentials), `SQS_QUEUE_URL`, and the R2 download credentials.
   An AWS instance role can replace explicit AWS keys. `SQS_REGION` is the input
   queue's region. Never use a queue consumed by the mask worker.
3. Publish models once as described above, or copy the already verified checkpoint
   tree to the machine's `checkpoints/` directory.
4. Build and start:

```bash
docker compose up --build -d worker
docker compose logs -f worker
```

Compose runs the model downloader first and mounts the persistent `checkpoints`
folder read-only in the GPU worker. The model loads once, verifies its tensors and
conditioner wiring, and then starts polling. One process serves jobs sequentially:
the underlying consumer mutates conditioning and RNG state per synthesis.

To use an already provisioned and verified checkpoint tree without the R2 init job:

```bash
docker compose build worker
docker compose run --rm --no-deps worker python -m voice_service.models verify
docker compose up -d --no-deps worker
```

`VOICE_DEPTH=fused` and `VOICE_COMPILE=false` are the defaults. All original depth
decoder choices are available as worker settings:

| `VOICE_DEPTH` | Behavior |
| --- | --- |
| `fused` | Runs the two guidance branches together; original consumer default. |
| `cached` | Incremental depth decoding with a KV cache. |
| `shipped` | Original upstream decoding loop, used for reference comparisons. |

These are execution options for the same model weights; they require no additional
checkpoint. Different decoders can produce different sampled audio with the same
seed, as described in the original inference documentation. `VOICE_COMPILE=true`
enables the original optional compilation path; initial capture is expensive.
`compile_requested` in metadata records the requested setting, not a guarantee that
capture succeeded. Restart the worker after changing these environment settings.

The input queue needs ReceiveMessage, DeleteMessage and ChangeMessageVisibility;
the result queue needs SendMessage. Media access needs S3 GetObject on references
and PutObject on outputs (plus KMS permissions if those objects use KMS). Publishing
models needs R2 read/write; deployment downloads need read only.

## Delivery behavior

The worker refreshes SQS visibility while a job is running. It uploads both outputs
and sends the terminal result **before deleting the request**. SIGTERM/SIGINT stops
polling and lets the current job finish; Compose allows 15 minutes before killing
the container. A forced stop leaves the request available for retry.

Invalid requests with usable routing fields and inference failures receive `failed`
results. Missing S3 input objects receive `failed` results. Cloud transport failures
retain the request for retry; malformed messages with no usable routing envelope
also remain on the queue. Configure a dead-letter queue and redrive limit so those
messages do not loop indefinitely, and monitor it in the backend/operations layer.

Delivery is **at least once**, not exactly once. A crash after publication but
before deletion can repeat a result. Backend consumers must deduplicate terminal
results by `fnf_job_id`. Retries write the same destination keys; use unique keys
for distinct jobs. No database or separate idempotency service is introduced.

## Local development and tests

Install [uv](https://docs.astral.sh/uv/) and run the lightweight suite:

```bash
uv sync --locked --only-group test
uv run --no-sync pytest
uv run --no-sync ruff check voice_service tests
```

These tests use emulated AWS and a synthesis double. They exercise S3 uploads,
metadata, SQS status/ack order, retries, reference routing, all three modes,
validation, float WAV preservation and atomic/checksummed model downloads. They
are not proof of voice quality or training-pipeline parity.

They also check per-job sampling overrides, restoration between jobs and decoder
selection. With the full inference dependencies installed, an additional CPU check
exercises the original consumer's sampling-configuration hook without loading
weights. That check is skipped with the lightweight test environment.

On a GPU host, install the full locked environment and render without queues:

```bash
uv sync --locked
uv run --no-sync python -m voice_service render examples/design.json --output outputs/design.wav
uv run --no-sync python -m voice_service render examples/clone.json \
  --reference speaker.wav --output outputs/clone.wav
uv run --no-sync python -m voice_service render examples/direction.json \
  --reference speaker.wav --output outputs/direction.wav
```

Local render validates the same payload but uses `--reference` and `--output`
instead of S3. It writes a metadata sidecar next to the WAV. The supplied research
parity scripts have additional training-repository dependencies described in
[the original documentation](docs/INFERENCE.md); those are not supplied here.

For a GPU check across all three modes in one process:

```bash
uv run --no-sync python scripts/gpu_smoke.py --reference speaker.wav
```

This writes listening examples and compares each wrapper output sample-for-sample
against a repeated direct call to the supplied consumer with the same parameters.
It checks wrapper fidelity and repeatability, not perceptual quality or parity
against the unavailable training pipeline. See [docs/VERIFICATION.md](docs/VERIFICATION.md)
for checks actually completed so far.

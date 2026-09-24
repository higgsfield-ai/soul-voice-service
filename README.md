# Soul Voice Service

SQS worker for voice design, cloning and direction, using the supplied Soul Voice
inference package and the same SQS/S3 pattern as Recast Mask Service.

## Voice task

`task_type=voice` generates one utterance. `voice_config.mode` selects:

1. `design`: generate a voice from a text instruction, without reference audio;
2. `clone`: use identity and delivery from a reference recording;
3. `direction`: use reference identity and blend reference delivery with the instruction.

`voice_config.checkpoint` selects `base`, `raft`, or `round1` (default). All three
checkpoints support every mode. They share the Breeze backbone, tokenizer and
Qwen audio codec on disk. Each checkpoint loads on first use and remains cached
in its own model stack on the GPU. One worker processes jobs sequentially.

Example design payload:

```json
{
  "job_id": "unique-job-id",
  "task_type": "voice",
  "s3_region": "eu-north-1",
  "sqs_region": "eu-north-1",
  "result_queue_url": "https://sqs.eu-north-1.amazonaws.com/ACCOUNT_ID/local-soul-voice-result",
  "dst_bucket_audio_pair": ["output-bucket", "voice/design.wav"],
  "voice_config": {
    "text": "We go live in thirty seconds.",
    "instruction": "A warm, calm female voice, speaking clearly and unhurriedly.",
    "mode": "design",
    "checkpoint": "round1",
    "seed": 4242
  }
}
```

For cloning or direction, change `mode` and add
`"src_bucket_audio_pair": ["input-bucket", "reference.wav"]`. The original
inference code converts references to mono 24 kHz and uses a seeded crop of up to
eight seconds. WAV and FLAC recordings readable by libsndfile are supported.

| Config field | Meaning and default |
| --- | --- |
| `text` | Required words to speak. |
| `instruction` | Required voice/delivery description in every mode. |
| `mode` | `design` (default), `clone`, or `direction`. |
| `checkpoint` | `base`, `raft`, or `round1` (default). |
| `seed` | Random seed, 0 through 2³²−1; default 0. |
| `style_mix_alpha` | Direction delivery: 0 uses the reference, 1 uses the instruction; default 1. Ignored by clone/design. |
| `temperature` | Positive backbone sampling temperature; default 0.9. |
| `top_k` | Backbone candidate-token cutoff; default 50. Zero disables it. |
| `depth_temperature` | Positive depth-decoder temperature; default 0.9. |
| `depth_top_k` | Depth-decoder candidate-token cutoff; default 50. Zero disables it. |
| `guidance_scale` | Instruction guidance strength; default 2.5. A value of 1 disables the guidance branch. |
| `max_new_tokens` | Positive generation limit for this job; defaults to `VOICE_MAX_NEW_TOKENS` (1024). |

Omitted or null sampling options use the consumer's defaults. Overrides apply only
to the current job, including after a failed generation. The environment token
limit is a default, not a hard ceiling. Long text can reach that limit; there is
no automatic sentence splitting. No voicebook was supplied, so optional prototype
voice casting is not enabled (`sample_voices=false`).

The worker uploads a mono, 24 kHz, float32 WAV to `dst_bucket_audio_pair`.
Metadata defaults to the same bucket with `.json` appended to the audio key;
override it with `dst_bucket_metadata_pair`. Input and output buckets are supplied
by each payload. Choose distinct output keys for audio, metadata and separate jobs.

The backend sends requests to `SQS_QUEUE_URL` and consumes `result_queue_url`.
The worker sends `in_progress`, followed by `completed` or `failed`, with the job
ID in `fnf_job_id`. Completion includes S3 URIs in `result_urls.audio` and
`result_urls.metadata`, `scores: null`, `bboxes: null`, and worker details in `meta`.
`meta.pipeline` contains the same JSON uploaded to S3: `schema_version`, `job_id`,
`task_type`, `voice_config`, exact `model_version`, effective `sampling`, `depth`,
`compile_requested`, `sample_voices`, reference URI, sample rate, channels, sample
count, duration, inference time and peak amplitude. Failure includes `fail_reason`.
The backend is responsible for routing jobs and delivering the results to users.

See [design](examples/design.json), [clone](examples/clone.json) and
[direction](examples/direction.json) payloads. Field definitions live in
[src/schemas/payload.py](src/schemas/payload.py) and
[src/schemas/voice.py](src/schemas/voice.py). Unknown fields are ignored, following
the other service's general payload behavior. Validation keeps voice-specific
requirements, without arbitrary text-length or routing-string limits.

## Local service

Copy `.env.example` to `.env` and fill in AWS credentials, including a session token
for temporary credentials. The same account provides SQS, S3 job-media and model
access. AWS roles/profiles can replace explicit keys. Never commit `.env`.

Set `SQS_QUEUE_URL` to `local-soul-voice` and use `local-soul-voice-result` for
payload results. Both queues are in `eu-north-1`. Set `VOICE_MODELS_HOST_DIR` to
the host checkpoint folder (default `./checkpoints`); both compose files mount it
at `/app/checkpoints`. `VOICE_CHECKPOINT_DIR` is the path used by Python inside the
container or during local execution.

Models are already published in the private S3 bucket `soul-voice-service`, under
`models/soul-voice/eb13a103470bdb26/`. The prefix contains `base/`, `raft/`, `round1/`
and `shared/`, about 10 GB in total. Configure these through `S3_MODEL_BUCKET`,
`S3_MODEL_REGION` and `S3_MODEL_PREFIX`.

Download all model weights:

```bash
docker compose -f docker-compose.download.yml up --build --force-recreate --exit-code-from download
```

`src.utils.download` syncs those four folders using `S3Client.sync_dir`. Existing
files are reused when their sizes and SHA-256 hashes match the S3 metadata.
Downloads replace local files only after verification; files uploaded without
SHA-256 metadata are downloaded again and checked by size. Download permissions
require S3 ListBucket and GetObject. A new model release should contain all four
folders under a new prefix; update `S3_MODEL_PREFIX` to select that release.

Start the worker:

```bash
docker compose up --build -d soul-voice-service
docker compose logs -f soul-voice-service
```

The GPU host needs Docker with NVIDIA Container Toolkit and a driver supporting
CUDA 13 (R580 or newer on Linux). The existing locked runtime is PyTorch 2.9.1+cu130,
Transformers 4.57.3 and qwen-tts 0.1.1. Models load locally with Hugging Face offline
mode enabled in the image. The supplied inference code and vendored Breeze source
are unchanged; see [docs/INFERENCE.md](docs/INFERENCE.md) and
[third_party/README.md](third_party/README.md).

`VOICE_DEPTH` accepts the original `fused` (default), `cached` and `shipped` decoder
implementations. `VOICE_COMPILE=true` enables the original optional compilation
path; default false. Different decoders can produce different sampled audio with
the same seed. Recreate the container after changing environment settings.
Checkpoint selection remains per payload and needs no restart.

Validate a payload without sending a message, then submit it:

```bash
uv run --no-sync python test_payload.py --payload examples/design.json --dry-run
uv run --no-sync python test_payload.py --payload examples/design.json --new-job-id
```

Replace example bucket names, account ID and output keys first. Upload reference
audio to the supplied S3 bucket/key for clone or direction. Queue access needs
ReceiveMessage, DeleteMessage and ChangeMessageVisibility on requests and
SendMessage on results. Media access needs GetObject for references and PutObject
for outputs.

The worker refreshes message visibility while processing and acknowledges only
after uploads and terminal result publication finish. Transient cloud failures
leave the request for retry; inference/validation failures with usable routing
receive a failed result. Malformed messages remain for retry/dead-letter handling.
Configure an SQS redrive limit. Results are at least once: the backend must
deduplicate by `fnf_job_id`. SIGTERM/SIGINT finishes the current job and stops
polling; Compose allows 15 minutes before forcing shutdown.

## Development

The service follows the Recast layout: `src/main.py`, `src/settings.py`, clients in
`src/clients`, queue handling in `src/internal`, inference in `src/core/pipeline.py`,
and request models in `src/schemas`. `soul_voice/` and `third_party/` retain the
original model implementations. Weights, recordings and credentials are not in Git.

Install the lightweight test environment and run the suite:

```bash
uv sync --locked --only-group test
uv run --no-sync pytest
uv run --no-sync ruff check src tests scripts test_payload.py test_voice_local.py
```

For local GPU inference, install the full environment and render without queues:

```bash
uv sync --locked
uv run --no-sync python test_voice_local.py --payload examples/design.json --output outputs/design.wav
uv run --no-sync python test_voice_local.py --payload examples/clone.json --reference speaker.wav --output outputs/clone.wav
uv run --no-sync python scripts/gpu_smoke.py --reference speaker.wav
```

The GPU helper renders all checkpoint/mode combinations and compares saved samples
with a direct call to the original consumer. Use `--checkpoints round1` to test one
version. These comparisons check wrapper fidelity, not perceptual quality or the
unavailable training pipeline. See [docs/VERIFICATION.md](docs/VERIFICATION.md) for
completed and pending checks. Preserve the supplied model licenses; Breeze's
weight license is included with the downloaded shared checkpoint.

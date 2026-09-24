import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    sqs_queue_url: str = os.getenv('SQS_QUEUE_URL', '')
    sqs_region: str = os.getenv('SQS_REGION', 'eu-north-1')
    sqs_visibility_timeout: int = int(os.getenv('SQS_VISIBILITY_TIMEOUT', '300'))
    aws_region: str = os.getenv('AWS_REGION', 'eu-north-1')
    s3_model_bucket: str = os.getenv('S3_MODEL_BUCKET', 'soul-voice-service')
    s3_model_region: str = os.getenv('S3_MODEL_REGION', os.getenv('AWS_REGION', 'eu-north-1'))
    s3_model_prefix: str = os.getenv('S3_MODEL_PREFIX', 'models/soul-voice/eb13a103470bdb26')
    checkpoints_dir: Path = Path(os.getenv('VOICE_CHECKPOINT_DIR', 'checkpoints'))
    source_dir: Path = Path(os.getenv('BREEZE_SOURCE_DIR', 'third_party/breeze-tts'))
    device: str = os.getenv('VOICE_DEVICE', 'cuda:0')
    depth: str = os.getenv('VOICE_DEPTH', 'fused')
    compile: bool = os.getenv('VOICE_COMPILE', 'false').lower() in ('true', '1')
    max_new_tokens: int = int(os.getenv('VOICE_MAX_NEW_TOKENS', '1024'))
    work_dir: Path = Path(os.getenv('WORK_DIR', '/tmp/soul-voice'))
    gpu_provider: str = os.getenv('GPU_PROVIDER', 'nebius')
    instance_name: str = os.getenv('INSTANCE_NAME', '')


settings = Settings()

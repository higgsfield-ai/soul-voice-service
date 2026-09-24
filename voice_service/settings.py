"""Environment settings; importing the service does not require credentials."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    queue_url: str = ""
    region: str = "eu-north-1"
    visibility_timeout: int = 300
    checkpoints_dir: Path = Path("checkpoints")
    source_dir: Path = Path("third_party/breeze-tts")
    device: str = "cuda:0"
    depth: str = "fused"
    compile: bool = False
    max_new_tokens: int = 1024
    work_dir: Path = Path("/tmp/soul-voice")
    gpu_provider: str = "nebius"
    instance_name: str = ""

    def __post_init__(self):
        if not 30 <= self.visibility_timeout <= 43200:
            raise ValueError("SQS_VISIBILITY_TIMEOUT must be between 30 and 43200 seconds")
        if self.depth not in {"shipped", "cached", "fused"}:
            raise ValueError("VOICE_DEPTH must be shipped, cached or fused")
        if self.max_new_tokens < 1:
            raise ValueError("VOICE_MAX_NEW_TOKENS must be positive")

    @classmethod
    def from_env(cls):
        load_dotenv()
        flag = os.getenv("VOICE_COMPILE", "false").lower()
        if flag not in {"true", "false", "1", "0"}:
            raise ValueError("VOICE_COMPILE must be true or false")
        return cls(
            queue_url=os.getenv("SQS_QUEUE_URL", ""),
            region=os.getenv("SQS_REGION", os.getenv("AWS_REGION", "eu-north-1")),
            visibility_timeout=int(os.getenv("SQS_VISIBILITY_TIMEOUT", "300")),
            checkpoints_dir=Path(os.getenv("VOICE_CHECKPOINT_DIR", "checkpoints")),
            source_dir=Path(os.getenv("BREEZE_SOURCE_DIR", "third_party/breeze-tts")),
            device=os.getenv("VOICE_DEVICE", "cuda:0"),
            depth=os.getenv("VOICE_DEPTH", "fused"),
            compile=flag in {"true", "1"},
            max_new_tokens=int(os.getenv("VOICE_MAX_NEW_TOKENS", "1024")),
            work_dir=Path(os.getenv("WORK_DIR", "/tmp/soul-voice")),
            gpu_provider=os.getenv("GPU_PROVIDER", "nebius"),
            instance_name=os.getenv("INSTANCE_NAME", ""),
        )

from typing import Literal

from pydantic import BaseModel, model_validator

from src.schemas.voice import VoiceConfig


class Payload(BaseModel):
    job_id: str
    s3_region: str
    sqs_region: str
    result_queue_url: str
    task_type: Literal['voice'] = 'voice'
    voice_config: VoiceConfig
    src_bucket_audio_pair: tuple[str, str] | None = None
    dst_bucket_audio_pair: tuple[str, str]
    dst_bucket_metadata_pair: tuple[str, str] | None = None

    @model_validator(mode='after')
    def validate(self):
        if self.voice_config.mode != 'design' and self.src_bucket_audio_pair is None:
            raise ValueError(f'{self.voice_config.mode} requires src_bucket_audio_pair')
        return self

    @property
    def metadata_location(self) -> tuple[str, str]:
        bucket, key = self.dst_bucket_audio_pair
        return self.dst_bucket_metadata_pair or (bucket, key + '.json')

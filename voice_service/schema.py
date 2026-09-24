"""Voice requests use the mask service's bucket/key pairs and routing fields."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]
S3Pair = tuple[NonEmpty, NonEmpty]


class VoiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    text: Annotated[str, Field(min_length=1, max_length=16000)]
    instruction: Annotated[str, Field(min_length=1, max_length=4000)]
    mode: Literal["design", "clone", "direction"] = "design"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    style_mix_alpha: float = Field(default=1.0, ge=0, le=1)

    # Omitted values inherit the original consumer's sampling defaults.
    temperature: float | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, ge=0)
    depth_temperature: float | None = Field(default=None, gt=0)
    depth_top_k: int | None = Field(default=None, ge=0)
    guidance_scale: float | None = None
    max_new_tokens: int | None = Field(default=None, gt=0)


class Envelope(BaseModel):
    # Unknown routing fields do not affect inference.
    job_id: NonEmpty
    s3_region: NonEmpty
    sqs_region: NonEmpty
    result_queue_url: NonEmpty


class Payload(Envelope):
    task_type: Literal["voice"]
    voice_config: VoiceConfig
    src_bucket_audio_pair: S3Pair | None = None
    dst_bucket_audio_pair: S3Pair
    dst_bucket_metadata_pair: S3Pair | None = None

    @model_validator(mode="after")
    def check_locations(self):
        if self.voice_config.mode != "design" and self.src_bucket_audio_pair is None:
            raise ValueError(f"{self.voice_config.mode} requires src_bucket_audio_pair")
        outputs = [self.dst_bucket_audio_pair, self.metadata_location]
        if len(set(outputs)) != 2:
            raise ValueError("audio and metadata must use different S3 objects")
        if self.src_bucket_audio_pair in outputs:
            raise ValueError("outputs must not overwrite the reference audio")
        return self

    @property
    def metadata_location(self) -> S3Pair:
        if self.dst_bucket_metadata_pair is not None:
            return self.dst_bucket_metadata_pair
        bucket, key = self.dst_bucket_audio_pair
        return bucket, key + ".json"


def s3_uri(pair: S3Pair) -> str:
    return f"s3://{pair[0]}/{pair[1]}"

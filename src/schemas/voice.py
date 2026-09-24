from typing import Literal

from pydantic import BaseModel, Field

Checkpoint = Literal['base', 'raft', 'round1']


class VoiceConfig(BaseModel):
    text: str
    instruction: str
    mode: Literal['design', 'clone', 'direction'] = 'design'
    checkpoint: Checkpoint = 'round1'
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    style_mix_alpha: float = Field(default=1.0, ge=0, le=1)

    # Omitted values use the original consumer's defaults.
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=0)
    depth_temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    depth_top_k: int | None = Field(default=None, ge=0)
    guidance_scale: float | None = Field(default=None, allow_inf_nan=False)
    max_new_tokens: int | None = Field(default=None, gt=0)

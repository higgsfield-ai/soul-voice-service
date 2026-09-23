"""Frozen Qwen3-TTS tokenizer frontend for the Stage-1 voice encoder.

The public ``Qwen3TTSTokenizerV2Model.encode`` only returns discrete codes, so
the pre-quantization path is re-run here from the same submodules. The frame
sequence produced below is byte-identical to what the quantizer consumes, which
``tests/test_voice_encoder.py`` pins by round-tripping to official codes.
"""

import math
from pathlib import Path

import torch
from torch import nn

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920
FRAME_RATE = SAMPLE_RATE / SAMPLES_PER_FRAME
FEATURE_SOURCES = ("tokenizer_prequant_latent", "tokenizer_encoder_hidden", "discrete_codec_embedding")
NUM_CODEBOOKS = 16


def frames_for_samples(samples: int | torch.Tensor) -> int | torch.Tensor:
    if isinstance(samples, torch.Tensor):
        return torch.div(samples + SAMPLES_PER_FRAME - 1, SAMPLES_PER_FRAME, rounding_mode="floor")
    return math.ceil(samples / SAMPLES_PER_FRAME)


class ReferenceFrontend(nn.Module):
    """Frozen acoustic frontend returning continuous features at 12.5 Hz."""

    def __init__(
        self,
        audio_tokenizer_dir: str | Path,
        *,
        feature_source: str = "tokenizer_prequant_latent",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if feature_source not in FEATURE_SOURCES:
            raise ValueError(f"feature_source must be one of {FEATURE_SOURCES}")
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Model,
        )

        directory = Path(audio_tokenizer_dir)
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"audio tokenizer config missing under {directory}")
        model = Qwen3TTSTokenizerV2Model.from_pretrained(directory, dtype=dtype)
        self.feature_source = feature_source
        # The waveform decoder half is never used by Stage 1.
        self.encoder = model.encoder.eval()
        self.encoder.requires_grad_(False)
        del model
        config = self.encoder.config
        if int(config.hidden_size) != 512:
            raise ValueError(f"unexpected tokenizer hidden size {config.hidden_size}")
        if float(config._frame_rate) != FRAME_RATE:
            raise ValueError(f"unexpected tokenizer frame rate {config._frame_rate}")
        self.output_dim = int(config.hidden_size)
        self.to(device)

    def train(self, mode: bool = True) -> "ReferenceFrontend":
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor, *, chunk_size: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``[B, N]`` audio into ``([B, T, 512], frame_lengths)``.

        The tokenizer stack is causal, so right padding never reaches the frames
        covered by ``lengths``. Rows are encoded in chunks because the convolution
        stack holds wide activations at waveform resolution, which would
        otherwise make peak memory scale with the whole batch.
        """
        if waveform.dim() != 2:
            raise ValueError(f"expected [B, N] waveform, got {tuple(waveform.shape)}")
        if chunk_size > 0 and waveform.shape[0] > chunk_size:
            features, frames = [], []
            for start in range(0, waveform.shape[0], chunk_size):
                window = slice(start, start + chunk_size)
                chunk, chunk_frames = self.forward(
                    waveform[window], lengths[window], chunk_size=0
                )
                features.append(chunk)
                frames.append(chunk_frames)
            return torch.cat(features), torch.cat(frames)

        parameter = next(self.encoder.parameters())
        values = waveform.to(device=parameter.device, dtype=parameter.dtype).unsqueeze(1)

        hidden = self.encoder.encoder(values)
        transformed = self.encoder.encoder_transformer(hidden.transpose(1, 2))[0]
        if self.feature_source == "tokenizer_encoder_hidden":
            features = transformed
            samples_per_frame = SAMPLES_PER_FRAME // 2
        else:
            embeddings = self.encoder.downsample(transformed.transpose(1, 2))
            if self.feature_source == "discrete_codec_embedding":
                codes = self.encoder.quantizer.encode(embeddings, NUM_CODEBOOKS)
                embeddings = self.encoder.quantizer.decode(codes.transpose(0, 1))
            features = embeddings.transpose(1, 2)
            samples_per_frame = SAMPLES_PER_FRAME

        counts = lengths.to(values.device) + samples_per_frame - 1
        frame_lengths = torch.div(counts, samples_per_frame, rounding_mode="floor")
        frame_lengths = frame_lengths.clamp(min=1, max=features.shape[1])
        return features.contiguous(), frame_lengths

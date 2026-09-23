"""Translate a service request into the existing, unmodified inference API."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from .schema import Payload
from .settings import Settings


class VoiceEngine:
    def __init__(self, settings: Settings):
        # Keep torch and the GPU model out of transport and download tools.
        import torch

        from soul_voice import Request, Sampling, VoiceConsumer

        if not settings.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Soul Voice inference requires an NVIDIA CUDA GPU")
        self.settings = settings
        self.request_type = Request
        self.consumer = VoiceConsumer.load(
            settings.bundle,
            source_dir=settings.source_dir,
            device=settings.device,
            sampling=Sampling(max_new_tokens=settings.max_new_tokens),
            depth=settings.depth,
            compile=settings.compile,
            strict=True,
        )
        self.gpu_type = torch.cuda.get_device_name(settings.device)

    def render(self, payload: Payload, reference: Path | None, output: Path) -> dict:
        config = payload.voice_config
        request = self.request_type(**config.model_dump(), reference=reference)
        start = perf_counter()
        (audio,) = self.consumer.synthesize([request], batch_size=1)
        inference_seconds = perf_counter() - start
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("model returned empty, non-finite or non-mono audio")
        # FLOAT preserves generated samples without PCM16 clipping or rounding.
        sf.write(output, audio, 24000, format="WAV", subtype="FLOAT")
        return {
            "schema_version": 1,
            "task_type": "voice",
            "voice_config": config.model_dump(mode="json"),
            "model_version": self.consumer.manifest["version"],
            "sampling": asdict(self.consumer.sampling),
            "depth": self.settings.depth,
            "compile_requested": self.settings.compile,
            "sample_voices": False,
            "sample_rate": 24000,
            "channels": 1,
            "samples": int(audio.size),
            "duration_seconds": audio.size / 24000,
            "inference_seconds": inference_seconds,
            "peak_amplitude": float(np.abs(audio).max()),
        }

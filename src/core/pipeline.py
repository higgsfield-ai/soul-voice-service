"""Translate a service request into the existing, unmodified inference API."""

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from src.schemas.voice import Checkpoint, VoiceConfig
from src.settings import Settings, settings


class Pipeline:
    def __init__(self, settings: Settings = settings):
        # Keep torch and the GPU model out of transport and download tools.
        import torch

        from soul_voice import Request, Sampling, VoiceConsumer

        if not settings.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Soul Voice inference requires an NVIDIA CUDA GPU')
        self.settings = settings
        self.request_type = Request
        self.consumer_type = VoiceConsumer
        self.sampling_type = Sampling
        self.consumers = {}
        self.gpu_type = torch.cuda.get_device_name(settings.device)

    def get_consumer(self, checkpoint: Checkpoint):
        # Jobs run sequentially. Cache only successfully loaded consumers, with
        # separate model and conditioning state for each checkpoint.
        if checkpoint not in self.consumers:
            self.consumers[checkpoint] = self.consumer_type.load(
                self.settings.checkpoints_dir / checkpoint,
                source_dir=self.settings.source_dir,
                device=self.settings.device,
                sampling=self.sampling_type(max_new_tokens=self.settings.max_new_tokens),
                depth=self.settings.depth,
                compile=self.settings.compile,
                strict=True,
            )
        return self.consumers[checkpoint]

    def __call__(self, reference: Path | None, output: Path, config: VoiceConfig) -> dict:
        values = config.model_dump()
        consumer = self.get_consumer(values.pop('checkpoint'))
        previous_sampling = consumer.sampling
        overrides = {name: values.pop(name) for name in asdict(previous_sampling)}
        sampling = replace(
            previous_sampling, **{name: value for name, value in overrides.items() if value is not None}
        )
        request = self.request_type(**values, reference=reference)
        start = perf_counter()
        # The worker calls this sequentially. Restore both the consumer settings
        # and the backbone/depth generation configs before processing another job.
        consumer.sampling = sampling
        try:
            consumer._apply_sampling()
            (audio,) = consumer.synthesize([request], batch_size=1)
        finally:
            consumer.sampling = previous_sampling
            consumer._apply_sampling()
        inference_seconds = perf_counter() - start
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError('model returned empty, non-finite or non-mono audio')
        # FLOAT preserves generated samples without PCM16 clipping or rounding.
        sf.write(output, audio, 24000, format='WAV', subtype='FLOAT')
        return {
            'schema_version': 1,
            'task_type': 'voice',
            'voice_config': config.model_dump(mode='json', exclude_none=True),
            'model_version': consumer.manifest['version'],
            'sampling': asdict(sampling),
            'depth': self.settings.depth,
            'compile_requested': self.settings.compile,
            'sample_voices': False,
            'sample_rate': 24000,
            'channels': 1,
            'samples': int(audio.size),
            'duration_seconds': audio.size / 24000,
            'inference_seconds': inference_seconds,
            'peak_amplitude': float(np.abs(audio).max()),
        }

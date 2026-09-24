import sys
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from voice_service.engine import VoiceEngine
from voice_service.schema import Payload
from voice_service.settings import Settings


@pytest.mark.parametrize("mode", ["design", "clone", "direction"])
def test_sampling_overrides_apply_only_to_current_job(payload, tmp_path, adapter, mode):
    payload["voice_config"]["mode"] = mode
    payload["src_bucket_audio_pair"] = ["voice-test-media", "reference.wav"]
    defaults = adapter.consumer.sampling
    overrides = {
        "temperature": 0.7,
        "top_k": 30,
        "depth_temperature": 0.8,
        "depth_top_k": 20,
        "guidance_scale": 1.0,
        "max_new_tokens": 128,
    }
    original = Payload.model_validate(payload)
    payload["voice_config"].update(overrides)
    observed = []

    def synthesize(*_args, **_kwargs):
        observed.append((asdict(adapter.consumer.sampling), adapter.consumer.applied_sampling.copy()))
        return [np.zeros(24, dtype=np.float32)]

    adapter.consumer.synthesize.side_effect = synthesize
    reference = None if mode == "design" else tmp_path / "reference.wav"
    metadata = adapter.render(Payload.model_validate(payload), reference, tmp_path / "custom.wav")
    assert observed == [(overrides, overrides)]
    assert metadata["sampling"] == overrides
    assert all(metadata["voice_config"][key] == value for key, value in overrides.items())
    assert adapter.consumer.sampling is defaults
    assert adapter.consumer.applied_sampling == asdict(defaults)
    # An ordinary request after an overridden one must still use the original defaults.
    metadata = adapter.render(original, reference, tmp_path / "default.wav")
    assert observed[-1] == (asdict(defaults), asdict(defaults))
    assert metadata["sampling"] == asdict(defaults)


def test_sampling_restored_after_inference_failure(payload, tmp_path, adapter):
    defaults = adapter.consumer.sampling
    payload["voice_config"].update(temperature=0.6, guidance_scale=1.0, max_new_tokens=64)
    adapter.consumer.synthesize.side_effect = RuntimeError("synthesis failed")
    with pytest.raises(RuntimeError, match="synthesis failed"):
        adapter.render(Payload.model_validate(payload), None, tmp_path / "failed.wav")
    assert adapter.consumer.sampling is defaults
    assert adapter.consumer.applied_sampling == asdict(defaults)


def test_partial_sampling_override_keeps_worker_defaults(payload, tmp_path, adapter):
    adapter.consumer.sampling = replace(adapter.consumer.sampling, max_new_tokens=256)
    payload["voice_config"].update(top_k=0, depth_top_k=0)
    metadata = adapter.render(Payload.model_validate(payload), None, tmp_path / "audio.wav")
    assert metadata["sampling"] == {
        "temperature": 0.9,
        "top_k": 0,
        "depth_temperature": 0.9,
        "depth_top_k": 0,
        "guidance_scale": 2.5,
        "max_new_tokens": 256,
    }


def test_null_sampling_values_use_defaults(payload, tmp_path, adapter):
    defaults = asdict(adapter.consumer.sampling)
    payload["voice_config"].update({name: None for name in defaults})
    metadata = adapter.render(Payload.model_validate(payload), None, tmp_path / "audio.wav")
    assert metadata["sampling"] == defaults


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0),
        ("temperature", -1),
        ("temperature", float("inf")),
        ("depth_temperature", 0),
        ("top_k", -1),
        ("depth_top_k", -1),
        ("guidance_scale", float("nan")),
        ("max_new_tokens", 0),
    ],
)
def test_invalid_sampling_values(payload, field, value):
    payload["voice_config"][field] = value
    with pytest.raises(ValidationError):
        Payload.model_validate(payload)


@pytest.mark.parametrize("depth", ["fused", "cached", "shipped"])
@pytest.mark.parametrize("compile", [False, True])
def test_all_decoder_options_reach_original_loader(monkeypatch, adapter, depth, compile):
    load = Mock(return_value=adapter.consumer)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test-gpu")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "soul_voice",
        SimpleNamespace(
            Request=adapter.request_type,
            Sampling=type(adapter.consumer.sampling),
            VoiceConsumer=SimpleNamespace(load=load),
        ),
    )
    monkeypatch.setenv("VOICE_DEPTH", depth)
    monkeypatch.setenv("VOICE_COMPILE", str(compile).lower())
    monkeypatch.setenv("VOICE_MAX_NEW_TOKENS", "256")
    engine = VoiceEngine(Settings.from_env())
    assert load.call_args.kwargs["depth"] == depth
    assert load.call_args.kwargs["compile"] is compile
    assert load.call_args.kwargs["sampling"].max_new_tokens == 256
    assert engine.settings.depth == depth


def test_sampling_reaches_original_generation_configs(payload, tmp_path, adapter, monkeypatch):
    # Optional with the lightweight test group; exercises the real consumer hook
    # when inference dependencies are installed, without loading weights or a GPU.
    pytest.importorskip("torch")
    from pathlib import Path
    from types import MethodType

    from soul_voice import Sampling, VoiceConsumer

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "third_party/breeze-tts"))
    consumer = adapter.consumer
    consumer.sampling = Sampling()
    consumer.model = SimpleNamespace(
        generation_config=SimpleNamespace(),
        depth_decoder=SimpleNamespace(generation_config=SimpleNamespace()),
    )
    consumer._apply_sampling = MethodType(VoiceConsumer._apply_sampling, consumer)
    snapshots = []

    def synthesize(*_args, **_kwargs):
        snapshots.append(
            (
                vars(consumer.model.generation_config).copy(),
                vars(consumer.model.depth_decoder.generation_config).copy(),
                asdict(consumer.sampling),
            )
        )
        return [np.zeros(24, dtype=np.float32)]

    consumer.synthesize.side_effect = synthesize
    payload["voice_config"].update(
        temperature=0.6, top_k=0, depth_temperature=0.7, depth_top_k=25, guidance_scale=1.0, max_new_tokens=64
    )
    metadata = adapter.render(Payload.model_validate(payload), None, tmp_path / "audio.wav")
    backbone, depth, active = snapshots[0]
    assert (backbone["temperature"], backbone["top_k"], backbone["max_new_tokens"]) == (0.6, 0, 64)
    assert (depth["temperature"], depth["top_k"]) == (0.7, 25)
    assert active["guidance_scale"] == 1.0
    assert metadata["sampling"] == active
    assert consumer.model.generation_config.temperature == Sampling().temperature
    assert consumer.model.depth_decoder.generation_config.top_k == Sampling().depth_top_k

import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.pipeline import Pipeline
from src.schemas import Payload
from src.settings import Settings


@pytest.fixture
def engine(monkeypatch, adapter, tmp_path):
    original = adapter.get_consumer('round1')

    def load(bundle, **kwargs):
        consumer = SimpleNamespace(
            sampling=kwargs['sampling'],
            manifest={'version': f'fixture-{bundle.name}'},
            synthesize=Mock(return_value=[np.zeros(24, dtype=np.float32)]),
            _apply_sampling=Mock(),
        )
        return consumer

    loader = Mock(side_effect=load)
    monkeypatch.setitem(
        sys.modules,
        'torch',
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'gpu')),
    )
    monkeypatch.setitem(
        sys.modules,
        'soul_voice',
        SimpleNamespace(
            Request=adapter.request_type,
            Sampling=type(original.sampling),
            VoiceConsumer=SimpleNamespace(load=loader),
        ),
    )
    engine = Pipeline(Settings(checkpoints_dir=tmp_path / 'checkpoints'))
    loader.assert_not_called()
    return engine


def test_omitted_checkpoint_defaults_to_round1(payload):
    assert Payload.model_validate(payload).voice_config.checkpoint == 'round1'


@pytest.mark.parametrize('checkpoint', ['unknown', '../raft', '/tmp/round1', None])
def test_checkpoint_is_a_supported_name(payload, checkpoint):
    payload['voice_config']['checkpoint'] = checkpoint
    with pytest.raises(ValidationError):
        Payload.model_validate(payload)


@pytest.mark.parametrize('mode', ['design', 'clone', 'direction'])
def test_alternating_checkpoints_reuse_consumers_and_preserve_sampling(engine, payload, tmp_path, mode):
    payload['voice_config']['mode'] = mode
    payload['src_bucket_audio_pair'] = ['voice-test-media', 'reference.wav']
    reference = None if mode == 'design' else tmp_path / 'reference.wav'
    observed = {}
    for checkpoint in ('round1', 'base', 'raft', 'round1', 'raft', 'base'):
        payload['voice_config'].update(checkpoint=checkpoint, temperature=0.6, max_new_tokens=64)
        meta = engine(reference, tmp_path / 'out.wav', config=(Payload.model_validate(payload)).voice_config)
        consumer = engine.get_consumer(checkpoint)
        assert meta['voice_config']['checkpoint'] == checkpoint
        assert meta['model_version'] == f'fixture-{checkpoint}'
        assert meta['sampling']['temperature'] == 0.6
        assert meta['sampling']['max_new_tokens'] == 64
        assert asdict(consumer.sampling) == asdict(engine.sampling_type())
        request = consumer.synthesize.call_args.args[0][0]
        assert request.mode == mode and request.reference == reference
        assert not hasattr(request, 'checkpoint')
        assert observed.setdefault(checkpoint, consumer) is consumer
    assert len({id(consumer) for consumer in observed.values()}) == 3
    assert engine.consumer_type.load.call_count == 3
    assert [call.args[0] for call in engine.consumer_type.load.call_args_list] == [
        engine.settings.checkpoints_dir / checkpoint for checkpoint in ('round1', 'base', 'raft')
    ]
    for call in engine.consumer_type.load.call_args_list:
        assert call.kwargs['strict'] is True
    # An old payload with no checkpoint or sampling fields still uses round1 defaults.
    for field in ('checkpoint', 'temperature', 'max_new_tokens'):
        payload['voice_config'].pop(field)
    meta = engine(reference, tmp_path / 'default.wav', config=(Payload.model_validate(payload)).voice_config)
    assert meta['model_version'] == 'fixture-round1'
    assert meta['sampling'] == asdict(engine.sampling_type())


def test_failed_checkpoint_load_can_retry_without_affecting_cached_model(engine):
    cached = engine.get_consumer('round1')
    load = engine.consumer_type.load
    factory = load.side_effect
    load.side_effect = RuntimeError('missing checkpoint')
    with pytest.raises(RuntimeError, match='missing checkpoint'):
        engine.get_consumer('raft')
    assert engine.consumers == {'round1': cached}
    load.side_effect = factory
    assert engine.get_consumer('raft').manifest['version'] == 'fixture-raft'
    assert engine.get_consumer('round1') is cached

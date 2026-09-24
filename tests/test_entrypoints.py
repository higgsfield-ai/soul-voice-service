import json
import signal
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import main
from src.settings import Settings


def test_main_wires_processor_consumer_and_shutdown(monkeypatch):
    config = Settings(sqs_queue_url='queue', sqs_region='eu-north-1')
    monkeypatch.setattr(main, 'settings', config)
    processor, consumer = Mock(), Mock()
    processor_factory = Mock(return_value=processor)
    consumer_factory = Mock(return_value=consumer)
    monkeypatch.setattr(main, 'MessageProcessor', processor_factory)
    monkeypatch.setattr(main, 'Consumer', consumer_factory)
    handlers = {}
    monkeypatch.setattr(main.signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))
    main.main()
    consumer_factory.assert_called_once_with('queue', 'eu-north-1', processor)
    consumer.run.assert_called_once_with()
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    consumer.stop.set.assert_called_once_with()


def test_missing_queue_does_not_load_models(monkeypatch):
    monkeypatch.setattr(main, 'settings', Settings(sqs_queue_url=''))
    processor = Mock()
    monkeypatch.setattr(main, 'MessageProcessor', processor)
    with pytest.raises(ValueError, match='SQS_QUEUE_URL'):
        main.main()
    processor.assert_not_called()


def test_payload_helper_dry_run_sends_nothing(payload, tmp_path, monkeypatch, capsys):
    import test_payload

    path = tmp_path / 'payload.json'
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, 'argv', ['test_payload.py', '--payload', str(path), '--dry-run'])
    client = Mock(side_effect=AssertionError('dry run must not contact AWS'))
    monkeypatch.setattr(test_payload.boto3, 'client', client)
    test_payload.main()
    assert json.loads(capsys.readouterr().out) == payload


def test_payload_helper_sends_to_request_queue(cloud, payload, tmp_path, monkeypatch):
    import test_payload

    path = tmp_path / 'payload.json'
    path.write_text(json.dumps(payload))
    monkeypatch.setenv('SQS_REGION', 'eu-north-1')
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'test_payload.py',
            '--payload',
            str(path),
            '--queue-url',
            cloud.requests,
            '--new-job-id',
        ],
    )
    test_payload.main()
    message = cloud.sqs.receive_message(QueueUrl=cloud.requests)['Messages'][0]
    sent = json.loads(message['Body'])
    assert sent['job_id'] != payload['job_id']
    assert sent['voice_config'] == payload['voice_config']


@pytest.mark.parametrize('mode', ['design', 'clone', 'direction'])
def test_local_render_uses_same_config_and_writes_metadata(payload, tmp_path, monkeypatch, mode):
    import test_voice_local

    reference = tmp_path / 'speaker.wav'
    reference.write_bytes(b'reference')
    output = tmp_path / 'out.wav'
    payload['voice_config'].update(mode=mode, checkpoint='raft')
    payload['src_bucket_audio_pair'] = ['input', 'reference.wav']
    path = tmp_path / 'payload.json'
    path.write_text(json.dumps(payload))
    calls = []

    def pipeline(source, destination, config):
        calls.append((source, destination, config))
        destination.write_bytes(b'generated')
        return {'voice_config': config.model_dump()}

    monkeypatch.setattr(test_voice_local, 'Pipeline', lambda: pipeline)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'test_voice_local.py',
            '--payload',
            str(path),
            '--reference',
            str(reference),
            '--output',
            str(output),
        ],
    )
    test_voice_local.main()
    assert calls[0][0] == (None if mode == 'design' else reference)
    assert calls[0][2].checkpoint == 'raft'
    assert output.read_bytes() == b'generated'
    metadata = json.loads(Path(str(output) + '.json').read_text())
    assert metadata['job_id'] == payload['job_id']
    assert metadata['voice_config']['mode'] == mode


def test_shutdown_releases_a_newly_received_message(cloud, monkeypatch):
    message = cloud.receive()

    def receive(timeout):
        cloud.worker.stop.set()
        return message

    monkeypatch.setattr(cloud.worker.sqs_client, 'receive_message', receive)
    cloud.worker.run()
    assert not cloud.engine.calls
    assert cloud.sqs.receive_message(QueueUrl=cloud.requests)['Messages']


def test_consumer_run_processes_one_message_then_stops(cloud, payload, monkeypatch):
    cloud.sqs.send_message(QueueUrl=cloud.requests, MessageBody=json.dumps(payload))
    original = cloud.worker.handle

    def handle(message):
        original(message)
        cloud.worker.stop.set()

    monkeypatch.setattr(cloud.worker, 'handle', handle)
    cloud.worker.run()
    assert [event['status'] for event in cloud.result_messages()] == ['in_progress', 'completed']

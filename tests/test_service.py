import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from botocore.exceptions import EndpointConnectionError
from pydantic import ValidationError

from voice_service.schema import Payload
from voice_service.storage import Storage
from voice_service.worker import VisibilityLease


@pytest.mark.parametrize("mode", ["design", "clone", "direction"])
def test_complete_round_trip(cloud, payload, mode):
    payload["voice_config"]["mode"] = mode
    if mode != "design":
        payload["src_bucket_audio_pair"] = ["voice-test-media", "input/reference.wav"]
        cloud.s3.put_object(Bucket="voice-test-media", Key="input/reference.wav", Body=b"reference-bytes")
    message = cloud.receive()
    assert cloud.worker.handle(message)
    events = cloud.result_messages()
    assert [event["status"] for event in events] == ["in_progress", "completed"]
    complete = events[1]
    assert complete["fnf_job_id"] == payload["job_id"]
    assert complete["result_urls"]["audio"] == "s3://voice-test-media/output/voice.wav"
    metadata = json.loads(
        cloud.s3.get_object(Bucket="voice-test-media", Key="output/voice.wav.json")["Body"].read()
    )
    assert complete["meta"]["pipeline"] == metadata
    assert metadata["voice_config"]["mode"] == mode
    assert cloud.engine.calls[0][1] == (None if mode == "design" else b"reference-bytes")
    assert (
        cloud.s3.head_object(Bucket="voice-test-media", Key="output/voice.wav")["ContentType"] == "audio/wav"
    )
    assert not list(cloud.worker.settings.work_dir.iterdir())
    attrs = cloud.sqs.get_queue_attributes(QueueUrl=cloud.requests, AttributeNames=["All"])["Attributes"]
    assert attrs["ApproximateNumberOfMessagesNotVisible"] == "0"
    assert attrs["ApproximateNumberOfMessages"] == "0"


@pytest.mark.parametrize("mode", ["clone", "direction"])
def test_missing_reference_is_terminal_without_inference(cloud, payload, mode):
    payload["voice_config"]["mode"] = mode
    assert cloud.worker.handle(cloud.receive())
    (result,) = cloud.result_messages()
    assert result["status"] == "failed"
    assert "src_bucket_audio_pair" in result["fail_reason"]
    assert not cloud.engine.calls


def test_unroutable_message_is_not_acknowledged(cloud):
    assert not cloud.worker.handle(cloud.receive({"text": "missing envelope"}))
    assert not cloud.result_messages()
    assert not cloud.engine.calls
    assert (
        cloud.sqs.get_queue_attributes(QueueUrl=cloud.requests, AttributeNames=["All"])["Attributes"][
            "ApproximateNumberOfMessagesNotVisible"
        ]
        == "1"
    )


def test_missing_s3_reference_fails_with_result(cloud, payload):
    payload["voice_config"]["mode"] = "clone"
    payload["src_bucket_audio_pair"] = ["voice-test-media", "does-not-exist.wav"]
    assert cloud.worker.handle(cloud.receive())
    assert cloud.result_messages()[-1]["status"] == "failed"
    assert not cloud.engine.calls


def test_inference_error_fails_without_uploading(cloud):
    cloud.engine.error = RuntimeError("synthesis failed")
    assert cloud.worker.handle(cloud.receive())
    assert cloud.result_messages()[-1]["status"] == "failed"
    assert not cloud.s3.list_objects_v2(Bucket="voice-test-media").get("Contents")
    assert not list(cloud.worker.settings.work_dir.iterdir())


@pytest.mark.parametrize("failure", ["download", "upload", "completion"])
def test_cloud_failure_never_acknowledges_request(cloud, payload, monkeypatch, failure):
    payload["voice_config"]["mode"] = "clone"
    payload["src_bucket_audio_pair"] = ["voice-test-media", "reference.wav"]
    cloud.s3.put_object(Bucket="voice-test-media", Key="reference.wav", Body=b"reference")
    error = EndpointConnectionError(endpoint_url="https://test.invalid")
    if failure in {"download", "upload"}:
        monkeypatch.setattr(Storage, failure, Mock(side_effect=error))
    else:
        original = cloud.worker._publish

        def publish(envelope, result):
            if result["status"] == "completed":
                raise error
            return original(envelope, result)

        monkeypatch.setattr(cloud.worker, "_publish", publish)
    with pytest.raises(EndpointConnectionError):
        cloud.worker.handle(cloud.receive())
    assert (
        cloud.sqs.get_queue_attributes(QueueUrl=cloud.requests, AttributeNames=["All"])["Attributes"][
            "ApproximateNumberOfMessagesNotVisible"
        ]
        == "1"
    )
    assert not any(result["status"] == "failed" for result in cloud.result_messages())
    assert not list(cloud.worker.settings.work_dir.iterdir())


def test_uploads_then_completion_then_delete(cloud, monkeypatch):
    events = []
    original_upload, original_publish, original_delete = (
        Storage.upload,
        cloud.worker._publish,
        cloud.worker.sqs.delete_message,
    )

    def upload(*args):
        original_upload(*args)
        events.append("upload")

    def publish(envelope, result):
        original_publish(envelope, result)
        events.append(result["status"])

    def delete(**kwargs):
        original_delete(**kwargs)
        events.append("delete")

    monkeypatch.setattr(Storage, "upload", upload)
    monkeypatch.setattr(cloud.worker, "_publish", publish)
    monkeypatch.setattr(cloud.worker.sqs, "delete_message", delete)
    cloud.worker.handle(cloud.receive())
    assert events == ["in_progress", "upload", "upload", "completed", "delete"]


def test_custom_metadata_location(cloud, payload):
    payload["dst_bucket_metadata_pair"] = ["voice-test-media", "metadata/custom.json"]
    cloud.worker.handle(cloud.receive())
    assert cloud.result_messages()[-1]["result_urls"]["metadata"].endswith("/metadata/custom.json")


def test_fifo_result_queue(cloud, payload):
    cloud.results = cloud.sqs.create_queue(QueueName="results.fifo", Attributes={"FifoQueue": "true"})[
        "QueueUrl"
    ]
    payload["result_queue_url"] = cloud.results
    cloud.worker.handle(cloud.receive())
    events = cloud.sqs.receive_message(QueueUrl=cloud.results, MaxNumberOfMessages=10)["Messages"]
    assert [json.loads(event["Body"])["status"] for event in events] == ["in_progress", "completed"]


def test_visibility_renewal_error_is_propagated():
    sqs = Mock()
    sqs.change_message_visibility.side_effect = [None, RuntimeError("lost connection")]
    with VisibilityLease(sqs, "queue", "receipt", 0.03) as lease:
        lease.thread.join(timeout=2)
        with pytest.raises(RuntimeError, match="renewal failed"):
            lease.check()
    assert not lease.thread.is_alive()


def test_lost_lease_does_not_publish_completion_or_ack(cloud, monkeypatch):
    monkeypatch.setattr(VisibilityLease, "check", Mock(side_effect=RuntimeError("lease lost")))
    delete = Mock()
    monkeypatch.setattr(cloud.worker.sqs, "delete_message", delete)
    with pytest.raises(RuntimeError, match="lease lost"):
        cloud.worker.handle(cloud.receive())
    assert [r["status"] for r in cloud.result_messages()] == ["in_progress"]
    delete.assert_not_called()


@pytest.mark.parametrize("mode", ["design", "clone", "direction"])
def test_adapter_preserves_config_and_float_samples(payload, tmp_path, mode, adapter):
    payload["voice_config"].update(mode=mode, style_mix_alpha=0.35)
    payload["src_bucket_audio_pair"] = ["voice-test-media", "reference.wav"]
    samples = np.array([-1.2, -0.5, 0, 0.6, 1.3], dtype=np.float32)

    engine = adapter
    engine.consumer.synthesize.return_value = [samples]
    reference = None if mode == "design" else Path("reference.wav")
    output = tmp_path / "audio.wav"
    metadata = engine.render(Payload.model_validate(payload), reference, output)
    request = engine.consumer.synthesize.call_args.args[0][0]
    assert vars(request) == {
        **Payload.model_validate(payload).voice_config.model_dump(exclude_none=True),
        "reference": reference,
    }
    decoded, rate = sf.read(output, dtype="float32")
    np.testing.assert_array_equal(decoded, samples)
    assert rate == 24000
    assert metadata["voice_config"]["style_mix_alpha"] == 0.35
    assert metadata["samples"] == len(samples)
    assert metadata["sampling"] == asdict(engine.consumer.sampling)


@pytest.mark.parametrize("samples", [[], [float("nan")], [[0.1, 0.2]]])
def test_adapter_rejects_broken_model_audio(payload, tmp_path, samples, adapter):
    engine = adapter
    engine.consumer.synthesize.return_value = [samples]
    with pytest.raises(RuntimeError, match="audio"):
        engine.render(Payload.model_validate(payload), None, tmp_path / "audio.wav")


@pytest.mark.parametrize(
    "mutation",
    [
        {"mode": "unsupported"},
        {"style_mix_alpha": 1.1},
        {"style_mix_alpha": float("nan")},
        {"seed": -1},
        {"seed": 2**32},
        {"text": ""},
        {"instructions": "typo"},
    ],
)
def test_invalid_inference_parameters(payload, mutation):
    payload["voice_config"].update(mutation)
    with pytest.raises(ValidationError):
        Payload.model_validate(payload)


def test_output_cannot_overwrite_reference(payload):
    payload["src_bucket_audio_pair"] = payload["dst_bucket_audio_pair"]
    with pytest.raises(ValidationError, match="overwrite"):
        Payload.model_validate(payload)


def test_audio_and_metadata_cannot_overwrite_each_other(payload):
    payload["dst_bucket_metadata_pair"] = payload["dst_bucket_audio_pair"]
    with pytest.raises(ValidationError, match="different"):
        Payload.model_validate(payload)

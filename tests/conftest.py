import json
from types import SimpleNamespace

import boto3
import numpy as np
import pytest
import soundfile as sf
from moto import mock_aws

from voice_service.settings import Settings
from voice_service.worker import Worker


@pytest.fixture
def payload():
    return {
        "job_id": "voice-test",
        "task_type": "voice",
        "s3_region": "eu-north-1",
        "sqs_region": "eu-north-1",
        "result_queue_url": "https://sqs.eu-north-1.amazonaws.com/123456789012/results",
        "dst_bucket_audio_pair": ["voice-test-media", "output/voice.wav"],
        "voice_config": {
            "text": "Hello from Soul Voice.",
            "instruction": "A calm, warm voice.",
            "mode": "design",
            "seed": 42,
        },
    }


class StubEngine:
    gpu_type = "test-double"

    def __init__(self):
        self.calls = []
        self.error = None

    def render(self, payload, reference, output):
        if self.error:
            raise self.error
        self.calls.append((payload, reference.read_bytes() if reference else None))
        sf.write(output, np.zeros(2400, dtype=np.float32), 24000, subtype="FLOAT")
        return {
            "inference_seconds": 0.1,
            "sample_rate": 24000,
            "samples": 2400,
            "voice_config": payload.voice_config.model_dump(),
            "model_version": "test-double",
        }


@pytest.fixture
def cloud(tmp_path, payload):
    # Real boto3 request/response shapes, emulated AWS; never contacts real queues.
    with mock_aws():
        session = boto3.Session(region_name="eu-north-1")
        sqs = session.client("sqs")
        requests = sqs.create_queue(QueueName="requests")["QueueUrl"]
        results = sqs.create_queue(QueueName="results")["QueueUrl"]
        s3 = session.client("s3")
        s3.create_bucket(
            Bucket="voice-test-media", CreateBucketConfiguration={"LocationConstraint": "eu-north-1"}
        )
        payload["result_queue_url"] = results
        engine = StubEngine()
        worker = Worker(Settings(queue_url=requests, work_dir=tmp_path / "work"), engine, session)

        def receive(body=None):
            sqs.send_message(QueueUrl=requests, MessageBody=json.dumps(payload if body is None else body))
            return sqs.receive_message(
                QueueUrl=requests, MessageSystemAttributeNames=["ApproximateReceiveCount"]
            )["Messages"][0]

        def result_messages():
            messages = sqs.receive_message(QueueUrl=results, MaxNumberOfMessages=10).get("Messages", [])
            return [json.loads(message["Body"]) for message in messages]

        yield SimpleNamespace(
            session=session,
            sqs=sqs,
            s3=s3,
            worker=worker,
            engine=engine,
            requests=requests,
            results=results,
            receive=receive,
            result_messages=result_messages,
        )

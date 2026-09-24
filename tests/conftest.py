import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import numpy as np
import pytest
import soundfile as sf
from moto import mock_aws

from src.core.pipeline import Pipeline
from src.internal.consumer import Consumer
from src.internal.message_processor import MessageProcessor
from src.settings import Settings


@dataclass
class StubSampling:
    temperature: float = 0.9
    top_k: int = 50
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    guidance_scale: float = 2.5
    max_new_tokens: int = 1024


@dataclass
class StubRequest:
    text: str
    instruction: str
    mode: str = 'design'
    reference: object = None
    seed: int = 0
    style_mix_alpha: float = 1.0


@pytest.fixture
def adapter():
    engine = Pipeline.__new__(Pipeline)
    engine.settings = Settings()
    engine.request_type = StubRequest
    consumer = SimpleNamespace(
        sampling=StubSampling(),
        manifest={'version': 'fixture'},
        synthesize=Mock(return_value=[np.zeros(24, dtype=np.float32)]),
    )

    def apply_sampling():
        consumer.applied_sampling = asdict(consumer.sampling)

    consumer._apply_sampling = Mock(side_effect=apply_sampling)
    apply_sampling()
    engine.consumers = {'round1': consumer}
    return engine


@pytest.fixture
def payload():
    return {
        'job_id': 'voice-test',
        'task_type': 'voice',
        's3_region': 'eu-north-1',
        'sqs_region': 'eu-north-1',
        'result_queue_url': 'https://sqs.eu-north-1.amazonaws.com/123456789012/results',
        'dst_bucket_audio_pair': ['voice-test-media', 'output/voice.wav'],
        'voice_config': {
            'text': 'Hello from Soul Voice.',
            'instruction': 'A calm, warm voice.',
            'mode': 'design',
            'seed': 42,
        },
    }


class StubEngine:
    gpu_type = 'test-double'

    def __init__(self):
        self.calls = []
        self.error = None

    def __call__(self, reference, output, config):
        if self.error:
            raise self.error
        self.calls.append((config, reference.read_bytes() if reference else None))
        sf.write(output, np.zeros(2400, dtype=np.float32), 24000, subtype='FLOAT')
        return {
            'inference_seconds': 0.1,
            'sample_rate': 24000,
            'samples': 2400,
            'voice_config': config.model_dump(),
            'model_version': 'test-double',
        }


@pytest.fixture
def cloud(tmp_path, payload):
    # Real boto3 request/response shapes, emulated AWS; never contacts real queues.
    with mock_aws():
        session = boto3.Session(region_name='eu-north-1')
        sqs = session.client('sqs')
        requests = sqs.create_queue(QueueName='requests')['QueueUrl']
        results = sqs.create_queue(QueueName='results')['QueueUrl']
        s3 = session.client('s3')
        s3.create_bucket(
            Bucket='voice-test-media', CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'}
        )
        payload['result_queue_url'] = results
        engine = StubEngine()
        config = Settings(sqs_queue_url=requests, work_dir=tmp_path / 'work')
        processor = MessageProcessor(engine, config)
        worker = Consumer(requests, config.sqs_region, processor, config.sqs_visibility_timeout)

        def receive(body=None):
            sqs.send_message(QueueUrl=requests, MessageBody=json.dumps(payload if body is None else body))
            return sqs.receive_message(
                QueueUrl=requests, MessageSystemAttributeNames=['ApproximateReceiveCount']
            )['Messages'][0]

        def result_messages():
            messages = sqs.receive_message(QueueUrl=results, MaxNumberOfMessages=10).get('Messages', [])
            return [json.loads(message['Body']) for message in messages]

        yield SimpleNamespace(
            session=session,
            sqs=sqs,
            s3=s3,
            worker=worker,
            processor=processor,
            engine=engine,
            requests=requests,
            results=results,
            receive=receive,
            result_messages=result_messages,
        )

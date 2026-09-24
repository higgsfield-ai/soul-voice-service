import hashlib
import json

import boto3
from botocore.config import Config

from src.settings import settings


class SqsClient:
    def __init__(self, queue_url: str, region: str = settings.sqs_region):
        self._queue_url = queue_url
        self._client = boto3.client(
            'sqs',
            region_name=region,
            config=Config(
                retries={'mode': 'standard', 'max_attempts': 3}, connect_timeout=10, read_timeout=30
            ),
        )

    def send_message(self, payload: dict):
        body = json.dumps(payload, allow_nan=False)
        kwargs = {'QueueUrl': self._queue_url, 'MessageBody': body}
        if self._queue_url.endswith('.fifo'):
            kwargs.update(
                MessageGroupId=hashlib.sha256(payload['fnf_job_id'].encode()).hexdigest(),
                MessageDeduplicationId=hashlib.sha256(body.encode()).hexdigest(),
            )
        self._client.send_message(**kwargs)

    def receive_message(self, visibility_timeout: int):
        response = self._client.receive_message(
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=visibility_timeout,
            MessageSystemAttributeNames=['ApproximateReceiveCount'],
        )
        messages = response.get('Messages', [])
        return messages[0] if messages else None

    def delete_message(self, message: dict):
        self._client.delete_message(QueueUrl=self._queue_url, ReceiptHandle=message['ReceiptHandle'])

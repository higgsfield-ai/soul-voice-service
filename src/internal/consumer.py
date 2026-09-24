import json
import logging
import threading

from pydantic import ValidationError

from src.clients.sqs import SqsClient
from src.internal.visibility import VisibilityLease
from src.schemas import Payload
from src.settings import settings

log = logging.getLogger(__name__)


class Consumer:
    def __init__(
        self,
        queue_url: str,
        sqs_region: str,
        message_processor,
        visibility_timeout: int = settings.sqs_visibility_timeout,
    ):
        self.queue_url = queue_url
        self.message_processor = message_processor
        self.visibility_timeout = visibility_timeout
        self.sqs_client = SqsClient(queue_url, sqs_region)
        self.stop = threading.Event()

    def handle(self, message: dict):
        try:
            body = json.loads(message['Body'])
        except (ValueError, TypeError):
            log.error('Invalid JSON; leaving request for retry/dead-letter queue')
            return False
        if not isinstance(body, dict) or not all(
            isinstance(body.get(key), str) and body[key]
            for key in ('job_id', 'sqs_region', 'result_queue_url')
        ):
            log.error('Missing result routing; leaving request for retry/dead-letter queue')
            return False

        retry_count = max(0, int(message.get('Attributes', {}).get('ApproximateReceiveCount', 1)) - 1)
        with VisibilityLease(
            self.sqs_client._client,
            self.queue_url,
            message['ReceiptHandle'],
            self.visibility_timeout,
        ) as lease:
            try:
                data = Payload.model_validate(body)
            except ValidationError as error:
                # Do not include the submitted text in validation errors.
                reason = '; '.join(
                    f'{".".join(map(str, item["loc"]))}: {item["msg"]}'
                    for item in error.errors(include_input=False, include_url=False)
                )[:2000]
                result = {
                    'fnf_job_id': body['job_id'],
                    'status': 'failed',
                    'fail_reason': reason,
                    'meta': self.message_processor._meta(self.queue_url, retry_count),
                }
            else:
                result = self.message_processor(data, self.queue_url, retry_count)
            lease.check()
            SqsClient(body['result_queue_url'], body['sqs_region']).send_message(result)
            lease.check()
        # Stop renewals before acknowledgement so they cannot race with deletion.
        lease.check()
        self.sqs_client.delete_message(message)
        log.info('Job %s: %s', body['job_id'], result['status'])
        return True

    def run(self):
        log.info('Consumer started: %s', self.queue_url)
        while not self.stop.is_set():
            try:
                message = self.sqs_client.receive_message(self.visibility_timeout)
                if message is None:
                    continue
                if self.stop.is_set():
                    self.sqs_client._client.change_message_visibility(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=message['ReceiptHandle'],
                        VisibilityTimeout=0,
                    )
                    break
                self.handle(message)
            except Exception:
                log.exception('Delivery interrupted; retaining request for retry')
                self.stop.wait(5)

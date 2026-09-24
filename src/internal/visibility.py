import logging
import threading
import time
from contextlib import AbstractContextManager

log = logging.getLogger(__name__)


class VisibilityLease(AbstractContextManager):
    """Extend visibility throughout generation, uploads and result publication."""

    def __init__(self, sqs, queue_url, receipt, timeout):
        self.sqs, self.queue_url, self.receipt = sqs, queue_url, receipt
        self.timeout = timeout
        self.stop = threading.Event()
        self.error = None
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _extend(self):
        # SQS limits visibility to 12 hours from receive, not from each extension.
        remaining = int(43190 - (time.monotonic() - self.started))
        if remaining <= 0:
            raise RuntimeError('SQS visibility lease reached its 12-hour limit')
        self.sqs.change_message_visibility(
            QueueUrl=self.queue_url,
            ReceiptHandle=self.receipt,
            VisibilityTimeout=min(self.timeout, remaining),
        )

    def _run(self):
        while not self.stop.wait(min(60, self.timeout / 3)):
            try:
                self._extend()
            except Exception as exc:
                self.error = exc
                log.exception('Lost SQS visibility lease; delivery will not be acknowledged')
                return

    def check(self):
        if self.error is not None:
            raise RuntimeError('SQS visibility renewal failed') from self.error

    def __enter__(self):
        self._extend()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

"""One job at a time. Acknowledge only after publishing a terminal result."""

import json
import logging
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import ValidationError

from .schema import Envelope, Payload, s3_uri
from .settings import Settings
from .storage import Storage

log = logging.getLogger(__name__)
AWS_CONFIG = Config(retries={"mode": "standard", "max_attempts": 3}, connect_timeout=10, read_timeout=30)


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
            raise RuntimeError("SQS visibility lease reached its 12-hour limit")
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
                log.exception("Lost SQS visibility lease; delivery will not be acknowledged")
                return

    def check(self):
        if self.error is not None:
            raise RuntimeError("SQS visibility renewal failed") from self.error

    def __enter__(self):
        self._extend()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def validation_reason(exc: ValidationError) -> str:
    # Exclude submitted text from validation errors.
    return "; ".join(
        f"{'.'.join(map(str, error['loc'])) or 'payload'}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )[:2000]


class Worker:
    def __init__(self, settings: Settings, engine, session=None):
        if not settings.queue_url:
            raise ValueError("SQS_QUEUE_URL is required")
        self.settings, self.engine = settings, engine
        self.session = session or boto3.Session()
        self.sqs = self.session.client("sqs", region_name=settings.region, config=AWS_CONFIG)
        self.stop = threading.Event()
        settings.work_dir.mkdir(parents=True, exist_ok=True)

    def _publish(self, envelope, result):
        sqs = self.session.client("sqs", region_name=envelope.sqs_region, config=AWS_CONFIG)
        kwargs = {"QueueUrl": envelope.result_queue_url, "MessageBody": json.dumps(result, allow_nan=False)}
        if envelope.result_queue_url.endswith(".fifo"):
            import hashlib

            kwargs["MessageGroupId"] = hashlib.sha256(envelope.job_id.encode()).hexdigest()
            kwargs["MessageDeduplicationId"] = hashlib.sha256(kwargs["MessageBody"].encode()).hexdigest()
        sqs.send_message(**kwargs)

    def _meta(self, message):
        return {
            "gpu_provider": self.settings.gpu_provider,
            "gpu_type": self.engine.gpu_type,
            "gpu_count": 1,
            "queue_url": self.settings.queue_url,
            "retry_count": max(0, int(message.get("Attributes", {}).get("ApproximateReceiveCount", 1)) - 1),
            "instance_name": self.settings.instance_name,
        }

    def _process(self, payload: Payload):
        storage = Storage(self.session.client("s3", region_name=payload.s3_region, config=AWS_CONFIG))
        with TemporaryDirectory(prefix="job-", dir=self.settings.work_dir) as directory:
            work = Path(directory)
            reference = None
            if payload.voice_config.mode != "design":
                reference = work / "reference.audio"
                # Cloud failures must escape to retry, not become ML failures.
                storage.download(payload.src_bucket_audio_pair, reference)
            output = work / "audio.wav"
            try:
                metadata = self.engine.render(payload, reference, output)
            except Exception as exc:
                raise InferenceFailure(f"{type(exc).__name__}: {exc}"[:2000]) from exc
            metadata.update(
                {
                    "job_id": payload.job_id,
                    "reference_audio": s3_uri(payload.src_bucket_audio_pair) if reference else None,
                }
            )
            metadata_file = work / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            storage.upload(output, payload.dst_bucket_audio_pair, "audio/wav")
            storage.upload(metadata_file, payload.metadata_location, "application/json")
        return metadata

    def handle(self, message):
        receipt = message["ReceiptHandle"]
        try:
            envelope = Envelope.model_validate_json(message["Body"])
        except ValidationError:
            log.error("Unroutable request %s; leaving for retry/dead-letter queue", message.get("MessageId"))
            return False

        with VisibilityLease(
            self.sqs, self.settings.queue_url, receipt, self.settings.visibility_timeout
        ) as lease:
            meta = self._meta(message)
            try:
                payload = Payload.model_validate_json(message["Body"])
            except ValidationError as exc:
                result = {
                    "fnf_job_id": envelope.job_id,
                    "status": "failed",
                    "fail_reason": validation_reason(exc),
                    "meta": meta,
                }
            else:
                self._publish(envelope, {"fnf_job_id": envelope.job_id, "status": "in_progress"})
                try:
                    metadata = self._process(payload)
                except InferenceFailure as exc:
                    log.exception("Voice inference failed for job %s", envelope.job_id)
                    result = {
                        "fnf_job_id": envelope.job_id,
                        "status": "failed",
                        "fail_reason": str(exc),
                        "meta": meta,
                    }
                except ClientError as exc:
                    if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NoSuchBucket"}:
                        raise
                    result = {
                        "fnf_job_id": envelope.job_id,
                        "status": "failed",
                        "fail_reason": "S3 input or output bucket/object does not exist",
                        "meta": meta,
                    }
                else:
                    meta.update(real_inference_time=metadata["inference_seconds"], pipeline=metadata)
                    result = {
                        "fnf_job_id": envelope.job_id,
                        "status": "completed",
                        "result_urls": {
                            "audio": s3_uri(payload.dst_bucket_audio_pair),
                            "metadata": s3_uri(payload.metadata_location),
                        },
                        "scores": None,
                        "bboxes": None,
                        "meta": meta,
                    }
            lease.check()
            self._publish(envelope, result)
            lease.check()
        # Stop heartbeat before deletion so a renewal cannot race with it.
        lease.check()
        self.sqs.delete_message(QueueUrl=self.settings.queue_url, ReceiptHandle=receipt)
        log.info("Job %s: %s", envelope.job_id, result["status"])
        return True

    def run(self):
        log.info("Voice worker ready; polling one job at a time")
        while not self.stop.is_set():
            try:
                response = self.sqs.receive_message(
                    QueueUrl=self.settings.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,
                    VisibilityTimeout=self.settings.visibility_timeout,
                    MessageSystemAttributeNames=["ApproximateReceiveCount"],
                )
                for message in response.get("Messages", []):
                    if self.stop.is_set():
                        self.sqs.change_message_visibility(
                            QueueUrl=self.settings.queue_url,
                            ReceiptHandle=message["ReceiptHandle"],
                            VisibilityTimeout=0,
                        )
                        break
                    try:
                        self.handle(message)
                    except Exception:
                        log.exception("Delivery interrupted; retaining request for retry")
            except Exception:
                log.exception("Queue polling failed")
                self.stop.wait(5)


class InferenceFailure(Exception):
    """Terminal synthesis failure, distinct from a retryable cloud failure."""

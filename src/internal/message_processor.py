import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from botocore.exceptions import ClientError

from src.clients.s3 import S3Client
from src.clients.sqs import SqsClient
from src.schemas import Payload
from src.settings import Settings, settings


def s3_uri(pair):
    return f's3://{pair[0]}/{pair[1]}'


log = logging.getLogger(__name__)


class MessageProcessor:
    def __init__(self, pipeline=None, settings: Settings = settings):
        if pipeline is None:
            from src.core.pipeline import Pipeline

            pipeline = Pipeline(settings)
        self.pipeline = pipeline
        self.settings = settings
        settings.work_dir.mkdir(parents=True, exist_ok=True)

    def _meta(self, queue_url: str, retry_count: int):
        return {
            'gpu_provider': self.settings.gpu_provider,
            'gpu_type': self.pipeline.gpu_type,
            'gpu_count': 1,
            'queue_url': queue_url,
            'retry_count': retry_count,
            'instance_name': self.settings.instance_name,
        }

    def _process(self, data: Payload):
        with TemporaryDirectory(prefix='job-', dir=self.settings.work_dir) as directory:
            work = Path(directory)
            reference = None
            if data.voice_config.mode != 'design':
                reference = work / 'reference.audio'
                bucket, key = data.src_bucket_audio_pair
                S3Client(bucket, data.s3_region).download_file(key, reference)
            output = work / 'audio.wav'
            try:
                metadata = self.pipeline(reference, output, config=data.voice_config)
            except Exception as error:
                raise InferenceFailure(f'{type(error).__name__}: {error}') from error
            metadata.update(
                job_id=data.job_id,
                reference_audio=s3_uri(data.src_bucket_audio_pair) if reference else None,
            )
            metadata_file = work / 'metadata.json'
            metadata_file.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
            for file, pair, content_type in (
                (output, data.dst_bucket_audio_pair, 'audio/wav'),
                (metadata_file, data.metadata_location, 'application/json'),
            ):
                S3Client(pair[0], data.s3_region).upload_file(file, pair[1], content_type)
        return metadata

    def __call__(self, data: Payload, queue_url: str, retry_count: int = 0):
        meta = self._meta(queue_url, retry_count)
        result = {'fnf_job_id': data.job_id, 'status': 'failed', 'meta': meta}
        SqsClient(data.result_queue_url, data.sqs_region).send_message(
            {'fnf_job_id': data.job_id, 'status': 'in_progress'},
        )
        try:
            metadata = self._process(data)
        except InferenceFailure as error:
            log.exception('Voice inference failed for job %s', data.job_id)
            result['fail_reason'] = str(error)[:2000]
        except ClientError as error:
            if error.response['Error']['Code'] not in {'404', 'NoSuchKey', 'NoSuchBucket'}:
                raise
            result['fail_reason'] = 'S3 input or output bucket/object does not exist'
        else:
            meta.update(real_inference_time=metadata['inference_seconds'], pipeline=metadata)
            result.update(
                status='completed',
                scores=None,
                bboxes=None,
                result_urls={
                    'audio': s3_uri(data.dst_bucket_audio_pair),
                    'metadata': s3_uri(data.metadata_location),
                },
            )
        return result


class InferenceFailure(Exception):
    pass

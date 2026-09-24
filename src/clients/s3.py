import hashlib
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
from botocore.config import Config

from src.settings import settings


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


log = logging.getLogger(__name__)


class S3Client:
    def __init__(self, bucket_name: str, region: str = settings.aws_region):
        self._bucket_name = bucket_name
        self._client = boto3.client(
            's3',
            region_name=region,
            config=Config(
                retries={'mode': 'standard', 'max_attempts': 3}, connect_timeout=10, read_timeout=30
            ),
        )

    def download_file(self, object_key: str, file_path: Path):
        log.info('Downloading s3://%s/%s', self._bucket_name, object_key)
        self._client.download_file(self._bucket_name, object_key, str(file_path))

    def upload_file(self, file_path: Path, object_key: str, content_type: str):
        self._client.upload_file(
            str(file_path),
            self._bucket_name,
            object_key,
            ExtraArgs={'ContentType': content_type},
        )

    def sync_dir(self, remote_dir: str, local_dir: Path):
        prefix = remote_dir.rstrip('/') + '/'
        root = Path(local_dir).resolve()
        count = 0
        pages = self._client.get_paginator('list_objects_v2').paginate(
            Bucket=self._bucket_name, Prefix=prefix
        )
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue
                target = (root / key[len(prefix) :]).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f'Model path escapes download directory: {key}')
                checksum = (
                    self._client.head_object(Bucket=self._bucket_name, Key=key)
                    .get('Metadata', {})
                    .get('sha256')
                )
                count += 1
                if (
                    checksum
                    and target.is_file()
                    and target.stat().st_size == obj['Size']
                    and sha256(target) == checksum
                ):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Never leave a partial download at the model's final path.
                with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    self.download_file(key, temporary)
                    if temporary.stat().st_size != obj['Size'] or (
                        checksum and sha256(temporary) != checksum
                    ):
                        raise ValueError(f'Model download is incomplete or corrupt: {key}')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        if not count:
            raise FileNotFoundError(f'No model files under s3://{self._bucket_name}/{prefix}')

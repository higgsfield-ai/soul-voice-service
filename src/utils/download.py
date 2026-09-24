import logging

from src.clients.s3 import S3Client
from src.settings import settings


def main():
    client = S3Client(settings.s3_model_bucket, settings.s3_model_region)
    for directory in ('base', 'raft', 'round1', 'shared'):
        client.sync_dir(
            f'{settings.s3_model_prefix}/{directory}',
            settings.checkpoints_dir / directory,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

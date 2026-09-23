"""Job media uses Amazon S3 and the normal AWS credential provider chain."""

from pathlib import Path

from .schema import S3Pair


class Storage:
    def __init__(self, client):
        self.client = client

    def download(self, location: S3Pair, destination: Path):
        self.client.download_file(location[0], location[1], str(destination))

    def upload(self, source: Path, location: S3Pair, content_type: str):
        self.client.upload_file(
            str(source),
            location[0],
            location[1],
            ExtraArgs={"ContentType": content_type},
        )

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from botocore.exceptions import EndpointConnectionError

from src.clients.s3 import S3Client
from src.settings import Settings
from src.utils import download


@pytest.fixture
def models(cloud):
    prefix = 'models/soul-voice/release'
    files = {
        'base/manifest.json': b'base-manifest',
        'raft/manifest.json': b'raft-manifest',
        'round1/manifest.json': b'round1-manifest',
        'shared/backbone/model.safetensors': b'backbone',
        'shared/base_checkpoint/audio_tokenizer/model.safetensors': b'codec',
    }
    for name, data in files.items():
        cloud.s3.put_object(
            Bucket='voice-test-media',
            Key=f'{prefix}/{name}',
            Body=data,
            Metadata={'sha256': hashlib.sha256(data).hexdigest()},
        )
    return prefix, files


def test_default_download_gets_all_bundles_and_shared_files(cloud, models, tmp_path, monkeypatch):
    prefix, files = models
    destination = tmp_path / 'checkpoints'
    config = Settings(s3_model_bucket='voice-test-media', s3_model_prefix=prefix, checkpoints_dir=destination)
    monkeypatch.setattr(download, 'settings', config)
    download.main()
    assert {
        p.relative_to(destination).as_posix(): p.read_bytes() for p in destination.rglob('*') if p.is_file()
    } == files
    # Matching files are reused, including when running download again at deployment.
    monkeypatch.setattr(S3Client, 'download_file', Mock(side_effect=AssertionError('already downloaded')))
    download.main()


def test_corrupt_cached_file_is_replaced(cloud, models, tmp_path):
    prefix, _ = models
    target = tmp_path / 'model.safetensors'
    target.write_bytes(b'bad-data')
    S3Client('voice-test-media').sync_dir(prefix + '/shared/backbone', tmp_path)
    assert target.read_bytes() == b'backbone'


@pytest.mark.parametrize('failure', ['checksum', 'interruption'])
def test_failed_download_keeps_previous_file(cloud, models, tmp_path, failure, monkeypatch):
    prefix, _ = models
    destination = tmp_path / 'download'
    destination.mkdir()
    target = destination / 'model.safetensors'
    target.write_bytes(b'old-data')
    client = S3Client('voice-test-media')

    def broken_download(key, destination):
        Path(destination).write_bytes(b'bad-data')
        if failure == 'interruption':
            raise EndpointConnectionError(endpoint_url='https://test.invalid')

    monkeypatch.setattr(client, 'download_file', broken_download)
    with pytest.raises((ValueError, EndpointConnectionError)):
        client.sync_dir(prefix + '/shared/backbone', destination)
    assert target.read_bytes() == b'old-data'
    assert list(destination.iterdir()) == [target]


def test_sync_uses_exact_directory_prefix_and_pagination(cloud, tmp_path):
    for key in ('models/base/a', 'models/base/nested/b', 'models/base-other/ignored'):
        cloud.s3.put_object(Bucket='voice-test-media', Key=key, Body=b'data')
    client = S3Client('voice-test-media')
    paginator = client._client.get_paginator('list_objects_v2')
    pages = paginator.paginate(
        Bucket='voice-test-media', Prefix='models/base/', PaginationConfig={'PageSize': 1}
    )
    client._client.get_paginator = Mock(return_value=Mock(paginate=Mock(return_value=pages)))
    client.sync_dir('models/base', tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()) == [
        'a',
        'nested/b',
    ]


def test_sync_supports_models_uploaded_without_hash_metadata(cloud, tmp_path):
    cloud.s3.put_object(Bucket='voice-test-media', Key='models/base/weights.pt', Body=b'weights')
    S3Client('voice-test-media').sync_dir('models/base', tmp_path)
    assert (tmp_path / 'weights.pt').read_bytes() == b'weights'


def test_missing_model_directory_fails(cloud, tmp_path):
    with pytest.raises(FileNotFoundError, match='No model files'):
        S3Client('voice-test-media').sync_dir('models/missing', tmp_path)


def test_sync_does_not_write_outside_checkpoint_directory(cloud, tmp_path):
    cloud.s3.put_object(Bucket='voice-test-media', Key='models/base/../escape.pt', Body=b'weights')
    with pytest.raises(ValueError, match='escapes'):
        S3Client('voice-test-media').sync_dir('models/base', tmp_path / 'base')
    assert not (tmp_path / 'escape.pt').exists()

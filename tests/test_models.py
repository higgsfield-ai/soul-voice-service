import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from botocore.exceptions import EndpointConnectionError
from pydantic import ValidationError

from voice_service.models import (
    Manifest,
    ModelFile,
    download,
    inventory,
    local_path,
    model_client,
    publish,
    verify,
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "source"
    for name in ("base", "raft", "round1", "shared/backbone", "shared/base_checkpoint/audio_tokenizer"):
        (root / name).mkdir(parents=True)
    for checkpoint in ("base", "raft", "round1"):
        (root / checkpoint / "manifest.json").write_text(
            json.dumps(
                {
                    "version": f"fixture-{checkpoint}",
                    "backbone": "../shared/backbone",
                    "base_checkpoint": "../shared/base_checkpoint",
                }
            )
        )
        (root / checkpoint / "voice_conditioner.pt").write_bytes(checkpoint.encode())
    (root / "shared/backbone/model.safetensors").write_bytes(b"backbone")
    (root / "shared/base_checkpoint/audio_tokenizer/model.safetensors").write_bytes(b"codec")
    return root, inventory(root)


def test_s3_publish_download_and_resume(cloud, bundle, tmp_path, monkeypatch):
    root, manifest = bundle
    publish(cloud.s3, "voice-test-media", manifest, root)
    destination = tmp_path / "download"
    download(cloud.s3, "voice-test-media", manifest, destination)
    verify(manifest, destination)
    copied_manifest = json.loads((destination / "round1/manifest.json").read_text())
    assert (destination / "round1" / copied_manifest["backbone"] / "model.safetensors").is_file()
    no_download = Mock(side_effect=AssertionError("unchanged file must be skipped"))
    monkeypatch.setattr(cloud.s3, "download_file", no_download)
    download(cloud.s3, "voice-test-media", manifest, destination)
    no_download.assert_not_called()
    no_upload = Mock(side_effect=AssertionError("unchanged file must be skipped"))
    monkeypatch.setattr(cloud.s3, "upload_file", no_upload)
    publish(cloud.s3, "voice-test-media", manifest, root)
    no_upload.assert_not_called()


def test_publish_refuses_changed_local_source(cloud, bundle):
    root, manifest = bundle
    (root / "round1/voice_conditioner.pt").write_bytes(b"different")
    with pytest.raises(ValueError, match="corrupt"):
        publish(cloud.s3, "voice-test-media", manifest, root)
    assert not cloud.s3.list_objects_v2(Bucket="voice-test-media").get("Contents")


def test_publish_refuses_remote_conflict(cloud, bundle):
    root, manifest = bundle
    cloud.s3.put_object(
        Bucket="voice-test-media", Key=f"{manifest.prefix}/{manifest.files[0].path}", Body=b"conflict"
    )
    with pytest.raises(ValueError, match="Refusing"):
        publish(cloud.s3, "voice-test-media", manifest, root)


@pytest.mark.parametrize("failure", ["checksum", "interruption"])
def test_failed_download_is_atomic(bundle, tmp_path, failure):
    _, manifest = bundle
    root = tmp_path / "destination"
    item = manifest.files[0]
    existing = root / item.path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old file")

    def broken_download(bucket, key, destination):
        Path(destination).write_bytes(b"partial bytes")
        if failure == "interruption":
            raise EndpointConnectionError(endpoint_url="https://s3.invalid")

    client = Mock(download_file=broken_download)
    with pytest.raises((ValueError, EndpointConnectionError)):
        download(client, "bucket", manifest, root)
    assert existing.read_bytes() == b"old file"
    assert not list(root.rglob(".download-*"))


@pytest.mark.parametrize("name", ["../outside", "/tmp/file", "foo/../../bar", "a\\b", "./foo", "."])
def test_inventory_rejects_unsafe_paths(name):
    with pytest.raises(ValidationError):
        ModelFile(path=name, size=0, sha256=hashlib.sha256(b"").hexdigest())


def test_download_rejects_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        local_path(root, "escape/file")


def test_duplicate_inventory_is_rejected(bundle):
    _, manifest = bundle
    with pytest.raises(ValidationError, match="duplicate"):
        Manifest(
            model_versions={"round1": "test"}, prefix="prefix", files=[manifest.files[0], manifest.files[0]]
        )


def test_release_prefix_changes_when_weights_change(bundle):
    root, before = bundle
    (root / "round1/voice_conditioner.pt").write_bytes(b"new weights")
    after = inventory(root)
    assert after.prefix != before.prefix


def test_inventory_includes_all_checkpoints_and_shared_files_once(bundle):
    _, manifest = bundle
    paths = [item.path for item in manifest.files]
    assert set(manifest.model_versions) == {"base", "raft", "round1"}
    for checkpoint in manifest.model_versions:
        assert f"{checkpoint}/manifest.json" in paths
        assert f"{checkpoint}/voice_conditioner.pt" in paths
    assert paths.count("shared/backbone/model.safetensors") == 1
    assert len(paths) == 8


def test_inventory_requires_every_supported_checkpoint(bundle):
    root, _ = bundle
    (root / "raft/manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        inventory(root)


def test_model_client_uses_aws_credentials_and_model_region(cloud, monkeypatch):
    monkeypatch.setenv("S3_MODEL_BUCKET", "voice-test-media")
    monkeypatch.setenv("S3_MODEL_REGION", "eu-north-1")
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://unused.invalid")
    client = model_client()
    assert client.meta.region_name == "eu-north-1"
    assert client.meta.endpoint_url == "https://s3.eu-north-1.amazonaws.com"
    client.put_object(Bucket="voice-test-media", Key="model-client-check", Body=b"aws")
    assert cloud.s3.get_object(Bucket="voice-test-media", Key="model-client-check")["Body"].read() == b"aws"


def test_model_client_requires_bucket(monkeypatch):
    monkeypatch.delenv("S3_MODEL_BUCKET", raising=False)
    with pytest.raises(ValueError, match="S3_MODEL_BUCKET"):
        model_client()

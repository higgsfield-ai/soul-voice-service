"""Publish and fetch the exact supplied model files using a checked-in inventory."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Literal

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class ModelFile(BaseModel):
    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def check_path(self):
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.path
            or str(path) != self.path
            or self.path == "."
        ):
            raise ValueError("model paths must be normalized, relative POSIX paths")
        return self


class Manifest(BaseModel):
    schema_version: Literal[1] = 1
    model_version: str
    prefix: str
    files: list[ModelFile] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_paths(self):
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate file in model inventory")
        return self


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("model path escapes the checkpoint directory")
    return path


def matches(path: Path, item: ModelFile) -> bool:
    return path.is_file() and path.stat().st_size == item.size and sha256(path) == item.sha256


def inventory(root: Path, bundle: str = "round1") -> Manifest:
    """Keep manifests' ../shared references valid when the tree is relocated."""
    bundle_path = local_path(root, bundle)
    bundle_manifest = json.loads((bundle_path / "manifest.json").read_text())
    directories = {bundle_path}
    for key in ("backbone", "base_checkpoint"):
        path = (bundle_path / bundle_manifest[key]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_dir():
            raise ValueError(f"{key} must point to an existing directory inside checkpoints")
        directories.add(path)
    files = []
    for directory in sorted(directories):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not any(
                part.startswith(".") for part in path.relative_to(root.resolve()).parts
            ):
                name = path.relative_to(root.resolve()).as_posix()
                actual = local_path(root, name)
                files.append(ModelFile(path=name, size=actual.stat().st_size, sha256=sha256(actual)))
    files.sort(key=lambda item: item.path)
    release = hashlib.sha256(
        json.dumps([f.model_dump() for f in files], sort_keys=True).encode()
    ).hexdigest()[:16]
    return Manifest(
        model_version=bundle_manifest["version"], prefix=f"models/soul-voice/{bundle}/{release}", files=files
    )


def verify(manifest: Manifest, root: Path):
    bad = [item.path for item in manifest.files if not matches(local_path(root, item.path), item)]
    if bad:
        raise ValueError("Missing or corrupt model files: " + ", ".join(bad))


def download(client, bucket: str, manifest: Manifest, root: Path):
    for item in manifest.files:
        target = local_path(root, item.path)
        if matches(target, item):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            client.download_file(bucket, f"{manifest.prefix}/{item.path}", str(temporary))
            if not matches(temporary, item):
                raise ValueError(f"Downloaded model checksum mismatch: {item.path}")
            temporary.replace(target)
            print(f"Downloaded {item.path}", flush=True)
        finally:
            temporary.unlink(missing_ok=True)


def publish(client, bucket: str, manifest: Manifest, root: Path):
    # Check every source before making any remote changes.
    verify(manifest, root)
    for item in manifest.files:
        key = f"{manifest.prefix}/{item.path}"
        try:
            existing = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if (
                existing.get("Metadata", {}).get("sha256") == item.sha256
                and existing["ContentLength"] == item.size
            ):
                continue
            raise ValueError(f"Refusing to replace different contents at {key}")
        client.upload_file(
            str(local_path(root, item.path)), bucket, key, ExtraArgs={"Metadata": {"sha256": item.sha256}}
        )
        print(f"Published {item.path}", flush=True)
    client.put_object(
        Bucket=bucket,
        Key=f"{manifest.prefix}/inventory.json",
        Body=manifest.model_dump_json(indent=2).encode(),
        ContentType="application/json",
    )


def r2_client():
    names = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise ValueError("Missing " + ", ".join(missing))
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", retries={"mode": "standard", "max_attempts": 3}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "verify", "publish", "download"])
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--manifest", type=Path, default=Path("models.json"))
    parser.add_argument("--bundle", default="round1", help="used only when generating an inventory")
    args = parser.parse_args()
    load_dotenv()
    if args.command == "inventory":
        manifest = inventory(args.root.resolve(), args.bundle)
        args.manifest.write_text(manifest.model_dump_json(indent=2) + "\n")
    else:
        manifest = Manifest.model_validate_json(args.manifest.read_text())
        if args.command == "verify":
            verify(manifest, args.root)
        else:
            client = r2_client()
            action = publish if args.command == "publish" else download
            action(client, os.environ["R2_BUCKET"], manifest, args.root)
    print(f"{args.command}: {len(manifest.files)} files, {sum(f.size for f in manifest.files):,} bytes")


if __name__ == "__main__":
    main()

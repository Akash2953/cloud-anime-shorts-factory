"""Cloudflare R2 storage client using S3-compatible boto3 interface."""

import io
import logging
from pathlib import Path

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


def build_client(account_id: str, access_key: str, secret_key: str) -> boto3.client:
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    cfg = Config(signature_version="s3v4", retries={"max_attempts": 3})
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=cfg,
    )


def upload_object(
    client: boto3.client,
    bucket: str,
    key: str,
    data: bytes | io.BytesIO,
    content_type: str = "application/octet-stream",
) -> str:
    extra = {}
    if isinstance(data, bytes):
        buf = io.BytesIO(data)
    else:
        buf = data
    client.put_object(Bucket=bucket, Key=key, Body=buf, ContentType=content_type)
    return f"{bucket}/{key}"


def upload_from_file(
    client: boto3.client,
    bucket: str,
    key: str,
    file_path: Path,
    content_type: str = "application/octet-stream",
) -> str:
    client.upload_file(str(file_path), bucket, key, ExtraArgs={"ContentType": content_type})
    return f"{bucket}/{key}"


def download_object(client: boto3.client, bucket: str, key: str) -> bytes:
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def presigned_url(
    client: boto3.client,
    bucket: str,
    key: str,
    expires_in: int = 3600,
) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )

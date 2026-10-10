"""
Cloudflare R2 client supporting both S3-compatible Boto3 mode
and automatic Cloudflare CLI / OAuth REST mode.
"""

from datetime import datetime
import io
import logging
from typing import Any, Dict, List, Optional, Tuple
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
import httpx

from app.config import get_cf_cli_oauth_token, settings
from app.image_utils import validate_object_key

logger = logging.getLogger(__name__)


class R2ClientError(Exception):
    """Base exception for R2 storage errors."""
    pass


class R2ConfigurationError(R2ClientError):
    """Raised when R2 credentials or configuration are missing."""
    pass


class R2ObjectNotFoundError(R2ClientError):
    """Raised when an object key is not found in the R2 bucket."""
    pass


class R2PayloadTooLargeError(R2ClientError):
    """Raised when a downloaded object exceeds safety byte limits."""
    pass


class R2Client:
    """
    Encapsulates safe operations on private Cloudflare R2 bucket.
    Automatically uses Cloudflare CLI OAuth token if S3 keys are not provided.
    """

    def __init__(self):
        self._s3_client = None

    @property
    def has_s3_credentials(self) -> bool:
        return bool(settings.r2_access_key_id and settings.r2_secret_access_key)

    @property
    def oauth_token(self) -> Optional[str]:
        return get_cf_cli_oauth_token()

    def _get_s3_client(self):
        """Lazy-initialize boto3 S3 client."""
        if self._s3_client is not None:
            return self._s3_client

        endpoint_url = settings.effective_r2_endpoint
        boto_config = Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=15,
        )

        self._s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region_name,
            config=boto_config,
        )
        return self._s3_client

    def check_connection(self) -> Dict[str, Any]:
        """Test connectivity to Cloudflare R2 bucket."""
        if not settings.is_r2_configured:
            return {
                "configured": False,
                "accessible": False,
                "message": "Cloudflare R2 credentials are not configured.",
            }

        # If S3 credentials are provided, test via Boto3
        if self.has_s3_credentials:
            try:
                client = self._get_s3_client()
                client.head_bucket(Bucket=settings.effective_bucket_name)
                return {
                    "configured": True,
                    "accessible": True,
                    "mode": "s3_boto3",
                    "bucket": settings.effective_bucket_name,
                    "message": "R2 S3 connection established successfully.",
                }
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code", "Unknown")
                return {
                    "configured": True,
                    "accessible": False,
                    "mode": "s3_boto3",
                    "bucket": settings.effective_bucket_name,
                    "error_code": code,
                    "message": f"R2 S3 client error: {code}",
                }
            except Exception as err:
                return {
                    "configured": True,
                    "accessible": False,
                    "mode": "s3_boto3",
                    "bucket": settings.effective_bucket_name,
                    "message": f"Connection check failed: {type(err).__name__}",
                }

        # Fall back to Cloudflare CLI OAuth token REST mode
        token = self.oauth_token
        if token:
            url = f"https://api.cloudflare.com/client/v4/accounts/{settings.effective_account_id}/r2/buckets/{settings.effective_bucket_name}/objects"
            headers = {"Authorization": f"Bearer {token}"}
            try:
                resp = httpx.get(url, headers=headers, params={"per_page": 1}, timeout=5.0)
                if resp.status_code == 200:
                    return {
                        "configured": True,
                        "accessible": True,
                        "mode": "cf_cli_oauth",
                        "bucket": settings.effective_bucket_name,
                        "message": "Connected via Cloudflare CLI authentication.",
                    }
                return {
                    "configured": True,
                    "accessible": False,
                    "mode": "cf_cli_oauth",
                    "bucket": settings.effective_bucket_name,
                    "message": f"Cloudflare API returned status {resp.status_code}",
                }
            except Exception as err:
                return {
                    "configured": True,
                    "accessible": False,
                    "mode": "cf_cli_oauth",
                    "bucket": settings.effective_bucket_name,
                    "message": f"REST check failed: {str(err)}",
                }

        return {"configured": False, "accessible": False, "message": "No credentials available."}

    def list_objects_paginated(
        self,
        prefix: str = "",
        max_keys: int = 100,
        continuation_token: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
        """List objects in bucket with pagination and prefix filtering."""
        if self.has_s3_credentials:
            client = self._get_s3_client()
            params = {
                "Bucket": settings.effective_bucket_name,
                "MaxKeys": max(1, min(max_keys, 1000)),
            }
            if prefix:
                params["Prefix"] = prefix.strip()
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            try:
                response = client.list_objects_v2(**params)
            except ClientError as err:
                raise R2ClientError("Failed to list objects in storage.") from err

            items = []
            for obj in response.get("Contents", []):
                items.append({
                    "key": obj["Key"],
                    "size_bytes": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "etag": obj["ETag"].strip('"'),
                })
            return items, response.get("NextContinuationToken"), response.get("IsTruncated", False)

        # Cloudflare REST API mode
        token = self.oauth_token
        if not token:
            raise R2ConfigurationError("No R2 credentials or Cloudflare CLI OAuth token found.")

        url = f"https://api.cloudflare.com/client/v4/accounts/{settings.effective_account_id}/r2/buckets/{settings.effective_bucket_name}/objects"
        params = {"per_page": max(1, min(max_keys, 500))}
        if prefix:
            params["prefix"] = prefix.strip()
        if continuation_token:
            params["cursor"] = continuation_token

        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = httpx.get(url, headers=headers, params=params, timeout=15.0)
            if resp.status_code != 200:
                raise R2ClientError(f"Cloudflare API error {resp.status_code}: {resp.text}")

            data = resp.json()
            raw_objects = data.get("result", [])
            info = data.get("result_info", {})

            items = []
            for obj in raw_objects:
                # Parse ISO date string
                lm_str = obj.get("last_modified")
                try:
                    lm_date = datetime.fromisoformat(lm_str.replace("Z", "+00:00")) if lm_str else datetime.utcnow()
                except Exception:
                    lm_date = datetime.utcnow()

                items.append({
                    "key": obj["key"],
                    "size_bytes": int(obj.get("size", 0)),
                    "last_modified": lm_date,
                    "etag": (obj.get("etag") or "").strip('"'),
                })

            next_cursor = info.get("cursor")
            is_truncated = bool(next_cursor)
            return items, next_cursor, is_truncated

        except httpx.RequestError as err:
            raise R2ClientError(f"Network error communicating with R2: {err}") from err

    def get_object_metadata(self, key: str) -> Dict[str, Any]:
        """Fetch object metadata without full download."""
        safe_key = validate_object_key(key)

        if self.has_s3_credentials:
            client = self._get_s3_client()
            try:
                resp = client.head_object(Bucket=settings.effective_bucket_name, Key=safe_key)
                return {
                    "key": safe_key,
                    "size_bytes": resp.get("ContentLength", 0),
                    "etag": resp.get("ETag", "").strip('"'),
                    "last_modified": resp.get("LastModified"),
                    "content_type": resp.get("ContentType", ""),
                }
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey"):
                    raise R2ObjectNotFoundError(f"Object '{safe_key}' does not exist.")
                raise R2ClientError(f"Failed to fetch metadata for '{safe_key}': {code}") from err

        # Cloudflare REST mode
        token = self.oauth_token
        url = f"https://api.cloudflare.com/client/v4/accounts/{settings.effective_account_id}/r2/buckets/{settings.effective_bucket_name}/objects/{safe_key}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0)
            if resp.status_code == 404:
                raise R2ObjectNotFoundError(f"Object '{safe_key}' does not exist in bucket.")
            if resp.status_code != 200:
                raise R2ClientError(f"Failed to fetch metadata for '{safe_key}': {resp.status_code}")

            etag = resp.headers.get("etag", "").strip('"')
            size_bytes = int(resp.headers.get("content-length", len(resp.content)))
            content_type = resp.headers.get("content-type", "")

            return {
                "key": safe_key,
                "size_bytes": size_bytes,
                "etag": etag,
                "last_modified": datetime.utcnow(),
                "content_type": content_type,
            }
        except httpx.RequestError as err:
            raise R2ClientError(f"Error connecting to Cloudflare R2: {err}") from err

    def download_object_bytes(
        self,
        key: str,
        max_bytes: int = settings.max_image_size_bytes,
    ) -> bytes:
        """Download object in chunks with strict byte limit safety check."""
        safe_key = validate_object_key(key)

        if self.has_s3_credentials:
            client = self._get_s3_client()
            try:
                resp = client.get_object(Bucket=settings.effective_bucket_name, Key=safe_key)
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey"):
                    raise R2ObjectNotFoundError(f"Object '{safe_key}' not found.")
                raise R2ClientError(f"Error fetching '{safe_key}': {code}") from err

            body = resp.get("Body")
            buffer = io.BytesIO()
            total = 0
            try:
                while True:
                    chunk = body.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise R2PayloadTooLargeError(f"Object '{safe_key}' exceeded {max_bytes} bytes.")
                    buffer.write(chunk)
            finally:
                body.close()
            return buffer.getvalue()

        # Cloudflare REST mode with chunked streaming
        token = self.oauth_token
        url = f"https://api.cloudflare.com/client/v4/accounts/{settings.effective_account_id}/r2/buckets/{settings.effective_bucket_name}/objects/{safe_key}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with httpx.stream("GET", url, headers=headers, timeout=20.0) as resp:
                if resp.status_code == 404:
                    raise R2ObjectNotFoundError(f"Object '{safe_key}' not found in R2.")
                if resp.status_code != 200:
                    raise R2ClientError(f"Failed to download '{safe_key}': status {resp.status_code}")

                buffer = io.BytesIO()
                total = 0
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise R2PayloadTooLargeError(f"Object '{safe_key}' exceeded {max_bytes} bytes.")
                    buffer.write(chunk)
                return buffer.getvalue()
        except httpx.RequestError as err:
            raise R2ClientError(f"Error streaming object '{safe_key}': {err}") from err

    def upload_object(
        self,
        key: str,
        data: bytes,
        content_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """Upload object to R2 bucket."""
        safe_key = validate_object_key(key)

        if self.has_s3_credentials:
            client = self._get_s3_client()
            try:
                resp = client.put_object(
                    Bucket=settings.effective_bucket_name,
                    Key=safe_key,
                    Body=data,
                    ContentType=content_type,
                )
                return {
                    "key": safe_key,
                    "etag": resp.get("ETag", "").strip('"'),
                    "size_bytes": len(data),
                }
            except ClientError as err:
                raise R2ClientError(f"Failed to upload '{safe_key}': {err}") from err

        # Cloudflare REST mode
        token = self.oauth_token
        url = f"https://api.cloudflare.com/client/v4/accounts/{settings.effective_account_id}/r2/buckets/{settings.effective_bucket_name}/objects/{safe_key}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        }
        try:
            resp = httpx.put(url, headers=headers, content=data, timeout=30.0)
            if resp.status_code not in (200, 201):
                raise R2ClientError(f"Upload failed: status {resp.status_code}: {resp.text}")

            res_json = resp.json().get("result", {})
            return {
                "key": safe_key,
                "etag": (res_json.get("etag") or "").strip('"'),
                "size_bytes": len(data),
            }
        except httpx.RequestError as err:
            raise R2ClientError(f"Failed to upload '{safe_key}': {err}") from err


# Global client instance
r2_client = R2Client()

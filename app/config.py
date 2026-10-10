"""Application configuration module using pydantic-settings."""

import json
from pathlib import Path
from typing import Any, List, Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_cf_cli_oauth_token() -> Optional[str]:
    """Retrieve active Cloudflare CLI OAuth token if authenticated."""
    token_path = Path.home() / "AppData/Roaming/xdg.config/cloudflare/config/default.json"
    if token_path.exists():
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("oauth_token")
        except Exception:
            pass
    return None


class Settings(BaseSettings):
    """Configuration loaded strictly from environment variables, .env file, or cf CLI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cloudflare R2 Credentials & Endpoint
    r2_account_id: Optional[str] = Field(default=None, alias="R2_ACCOUNT_ID")
    r2_access_key_id: Optional[str] = Field(default=None, alias="R2_ACCESS_KEY_ID")
    r2_secret_access_key: Optional[str] = Field(default=None, alias="R2_SECRET_ACCESS_KEY")
    r2_bucket_name: Optional[str] = Field(default=None, alias="R2_BUCKET_NAME")
    r2_endpoint_url: Optional[str] = Field(default=None, alias="R2_ENDPOINT_URL")
    r2_region_name: str = Field(default="auto", alias="R2_REGION_NAME")

    # Local SQLite Index Store
    sqlite_db_path: str = Field(default="data/index.db", alias="SQLITE_DB_PATH")

    # Face Engine Configuration
    face_engine_type: str = Field(default="insightface", alias="FACE_ENGINE_TYPE")
    insightface_model_name: str = Field(default="buffalo_s", alias="INSIGHTFACE_MODEL_NAME")
    match_similarity_threshold: float = Field(default=0.40, alias="MATCH_SIMILARITY_THRESHOLD")

    # Safety & Constraints
    max_image_size_bytes: int = Field(default=15 * 1024 * 1024, alias="MAX_IMAGE_SIZE_BYTES")  # 15MB
    max_image_dimension: int = Field(default=4096, alias="MAX_IMAGE_DIMENSION")
    min_image_dimension: int = Field(default=32, alias="MIN_IMAGE_DIMENSION")
    max_batch_concurrency: int = Field(default=2, alias="MAX_BATCH_CONCURRENCY")

    # Server & Networking
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    cors_origins_raw: Any = Field(
        default=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000", "http://127.0.0.1:3000"],
        alias="CORS_ORIGINS",
    )

    @property
    def cors_origins(self) -> List[str]:
        val = self.cors_origins_raw
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            if val == "*":
                return ["*"]
            if val.startswith("[") and val.endswith("]"):
                try:
                    return json.loads(val)
                except Exception:
                    pass
            return [origin.strip() for origin in val.split(",") if origin.strip()]
        return ["*"]

    @property
    def effective_account_id(self) -> str:
        """Effective Cloudflare account ID."""
        return self.r2_account_id or "c5b183bb011ad958f341adda9e499c13"

    @property
    def effective_bucket_name(self) -> str:
        """Effective R2 bucket name."""
        return self.r2_bucket_name or "purusharth-studio"

    @property
    def effective_r2_endpoint(self) -> str:
        """Compute the S3 endpoint URL for Cloudflare R2."""
        if self.r2_endpoint_url:
            return self.r2_endpoint_url.rstrip("/")
        return f"https://{self.effective_account_id}.r2.cloudflarestorage.com"

    @property
    def is_r2_configured(self) -> bool:
        """Check if S3 credentials or Cloudflare CLI OAuth token is available."""
        has_s3 = bool(
            self.r2_access_key_id
            and self.r2_secret_access_key
            and self.effective_bucket_name
            and (self.r2_account_id or self.r2_endpoint_url)
        )
        has_cf_cli = bool(get_cf_cli_oauth_token())
        return has_s3 or has_cf_cli

    def sanitized_dict(self) -> dict:
        """Return configuration safe for public exposure without sensitive credentials."""
        return {
            "r2_configured": self.is_r2_configured,
            "r2_bucket_name": self.effective_bucket_name,
            "r2_endpoint_url": self.effective_r2_endpoint,
            "sqlite_db_path": self.sqlite_db_path,
            "face_engine_type": self.face_engine_type,
            "match_similarity_threshold": self.match_similarity_threshold,
            "max_image_size_bytes": self.max_image_size_bytes,
            "environment": self.environment,
        }


# Global settings singleton
settings = Settings()

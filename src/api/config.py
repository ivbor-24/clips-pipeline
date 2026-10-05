"""API settings: ``API_*`` environment variables, then the project's ``.env``.

Real environment variables win over ``.env``, so Docker Compose and the shell
can override the file.
"""

from typing import Optional

from pydantic import Field, SecretStr, model_validator

from src.api.db_base import DatabaseSettings

# Values shipped in .env.example; accepting them would make tokens forgeable.
PLACEHOLDER_SECRETS = {"change-me-in-production"}

GENERATE_SECRET_HINT = 'python3 -c "import secrets; print(secrets.token_urlsafe(32))"'


class APISettings(DatabaseSettings):
    # Shared password. Empty: no login, the single built-in user
    # is always signed in. Set it before opening the service to the network.
    password: Optional[SecretStr] = None
    # Signs session tokens; needed only when a password is set.
    jwt_secret: Optional[SecretStr] = None
    jwt_algorithm: str = "HS256"
    session_days: int = Field(30, ge=1)
    # Largest video the web UI accepts, in GB (10**9 bytes).
    max_upload_gb: float = Field(10, gt=0)
    # Shut down after this many minutes without open web pages and without
    # queued or running jobs; 0 turns it off.
    idle_shutdown_min: int = Field(60, ge=0)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost"
    # Without a password the API answers only to localhost, IP addresses and
    # these host names (comma-separated): protection from DNS rebinding, see
    # src/api/security.py. Add the name you open the web UI by, if any.
    allowed_hosts: str = ""
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_host_list(self) -> list:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]

    @property
    def password_required(self) -> bool:
        return bool(self.password and self.password.get_secret_value())

    @model_validator(mode="after")
    def _require_secret_with_password(self) -> "APISettings":
        if not self.password_required:
            return self
        secret = self.jwt_secret.get_secret_value() if self.jwt_secret else ""
        if len(secret) < 16 or secret in PLACEHOLDER_SECRETS:
            raise ValueError(
                "API_PASSWORD is set, so API_JWT_SECRET must be a random string of at "
                f"least 16 characters. Generate one with: {GENERATE_SECRET_HINT}"
            )
        return self


settings = APISettings()

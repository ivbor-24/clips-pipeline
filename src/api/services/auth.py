"""Access to the API: one built-in user and an optional shared password.

Without ``API_PASSWORD`` every request acts as the built-in user. With it, the
password is exchanged for a session token signed with ``API_JWT_SECRET``.
"""

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from src.api.config import settings

# Owner of every job. Older databases had per-user accounts; their jobs are
# reassigned to this id at startup (see src.api.database.create_tables).
LOCAL_USER_ID = 1
SESSION_TOKEN_TYPE = "session"


@dataclass(frozen=True)
class LocalUser:
    """The single user of an installation."""

    id: int = LOCAL_USER_ID
    username: str = "local"


LOCAL_USER = LocalUser()


def _secret() -> str:
    return settings.jwt_secret.get_secret_value() if settings.jwt_secret else ""


def _password() -> str:
    return settings.password.get_secret_value() if settings.password else ""


def _password_fingerprint() -> str:
    """Tie tokens to the current password: changing it signs everyone out."""
    return hmac.new(_secret().encode(), _password().encode(), hashlib.sha256).hexdigest()[:16]


def check_password(candidate: str) -> bool:
    expected = _password()
    return bool(expected) and hmac.compare_digest(candidate.encode(), expected.encode())


def create_session_token() -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.session_days)
    claims = {
        "sub": LOCAL_USER.username,
        "type": SESSION_TOKEN_TYPE,
        "pw": _password_fingerprint(),
        "exp": expire,
    }
    return jwt.encode(claims, _secret(), algorithm=settings.jwt_algorithm)


def verify_session_token(token: str) -> bool:
    try:
        payload = jwt.decode(token, _secret(), algorithms=[settings.jwt_algorithm])
    except JWTError:
        return False
    return payload.get("type") == SESSION_TOKEN_TYPE and hmac.compare_digest(
        str(payload.get("pw", "")), _password_fingerprint()
    )

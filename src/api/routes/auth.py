from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.config import settings
from src.api.schemas.auth import LoginRequest, SessionResponse, TokenResponse
from src.api.services.auth import (
    LOCAL_USER,
    LocalUser,
    check_password,
    create_session_token,
    verify_session_token,
)

router = APIRouter()
logger = structlog.get_logger("auth")
bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    token: Optional[str] = Query(
        None, description="Session token for EventSource and <video>, which cannot send headers"
    ),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> LocalUser:
    if not settings.password_required:
        return LOCAL_USER
    if credentials:
        token = credentials.credentials
    if not token or not verify_session_token(token):
        raise HTTPException(
            status_code=401, detail="Password required", headers={"WWW-Authenticate": "Bearer"}
        )
    return LOCAL_USER


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request):
    if not settings.password_required:
        raise HTTPException(status_code=400, detail="Login is off: API_PASSWORD is not set")
    if not check_password(body.password):
        client = request.client.host if request.client else None
        logger.warning("login_failed", client=client)
        raise HTTPException(status_code=401, detail="Wrong password")
    return TokenResponse(access_token=create_session_token())


@router.get("/me", response_model=SessionResponse)
async def get_me(user: LocalUser = Depends(get_current_user)):
    return SessionResponse(username=user.username, password_required=settings.password_required)

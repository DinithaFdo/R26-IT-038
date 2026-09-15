from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime
from time import monotonic
from typing import Any

import httpx
from fastapi import Depends, Header, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.principals import (
    principal_from_clerk_claims,
    user_profile_from_clerk_claims,
)
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings, settings
from app.core.exceptions import AuthenticationError
from app.repositories.mongodb import MongoUserRepository
from app.repositories.protocols import UserRepository

logger = logging.getLogger(__name__)

SUPPORTED_HMAC_ALGORITHMS = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}

_jwks_cache = None
_jwks_refresh_lock = asyncio.Lock()

# This dependency exists primarily to describe Clerk authentication accurately
# in OpenAPI.  Runtime validation remains in ``require_authenticated_principal``
# below, which preserves the application's sanitised authentication errors.
# Swagger UI then renders an Authorize button and attaches the bearer token to
# every route that depends on ``require_clerk_user``.
clerk_bearer_scheme = HTTPBearer(
    scheme_name="ClerkBearerAuth",
    description="Paste a current Clerk session token. Swagger adds the Bearer prefix.",
    auto_error=False,
)


class ClerkJwksCache:
    def __init__(self, app_settings: Settings = settings) -> None:
        self._settings = app_settings
        self._jwks: dict[str, Any] | None = None
        self._expires_at = 0.0
        self._last_unknown_kid_refresh_at = float("-inf")
        self._negative_kid_cache: dict[str, float] = {}

    async def get_key(self, key_id: str) -> dict[str, Any]:
        jwks = await self.get_jwks()
        key = _find_jwk(jwks, key_id)
        if key is not None:
            return key

        if self._is_negative_kid_cached(key_id):
            raise AuthenticationError("Unknown Clerk signing key.", category="jwks_key_not_found")

        async with _jwks_refresh_lock:
            jwks = self._jwks or {}
            key = _find_jwk(jwks, key_id)
            if key is not None:
                return key
            if self._is_negative_kid_cached(key_id):
                raise AuthenticationError("Unknown Clerk signing key.", category="jwks_key_not_found")

            now = monotonic()
            minimum_interval = self._settings.clerk_jwks_min_refresh_interval_seconds
            if (
                now - self._last_unknown_kid_refresh_at
                < minimum_interval
            ):
                self._cache_negative_kid(key_id)
                raise AuthenticationError("Unknown Clerk signing key.", category="jwks_key_not_found")

            self._last_unknown_kid_refresh_at = now
            jwks = await self._refresh_unlocked()

        key = _find_jwk(jwks, key_id)
        if key is None:
            self._cache_negative_kid(key_id)
            raise AuthenticationError("Unknown Clerk signing key.", category="jwks_key_not_found")
        return key

    async def get_jwks(self) -> dict[str, Any]:
        if self._jwks is not None and monotonic() < self._expires_at:
            return self._jwks
        return await self.refresh()

    async def refresh(self) -> dict[str, Any]:
        async with _jwks_refresh_lock:
            if self._jwks is not None and monotonic() < self._expires_at:
                return self._jwks
            return await self._refresh_unlocked()

    async def _refresh_unlocked(self) -> dict[str, Any]:
        if not self._settings.clerk_jwks_url.strip():
            # The single most common local/dev misconfiguration: CLERK_ISSUER
            # and CLERK_JWKS_URL unset. This rejects every request before any
            # token is even inspected, so it is worth its own category rather
            # than folding into "jwks_fetch_failed".
            raise AuthenticationError(
                "Clerk JWKS URL is not configured.",
                category="missing_jwks_configuration",
            )

        try:
            jwks = await self._fetch_jwks()
        except AuthenticationError:
            raise
        except Exception as error:
            logger.warning("Clerk JWKS refresh failed.")
            raise AuthenticationError(
                "Clerk signing keys are unavailable.", category="jwks_fetch_failed"
            ) from error

        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise AuthenticationError(
                "Clerk JWKS payload is invalid.", category="jwks_fetch_failed"
            )

        self._jwks = jwks
        self._expires_at = monotonic() + self._settings.clerk_jwks_cache_seconds
        return jwks

    async def _fetch_jwks(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(self._settings.clerk_jwks_url)
            response.raise_for_status()
            return response.json()

    def _is_negative_kid_cached(self, key_id: str) -> bool:
        self._prune_negative_kid_cache()
        expires_at = self._negative_kid_cache.get(_kid_cache_key(key_id))
        return expires_at is not None and monotonic() < expires_at

    def _cache_negative_kid(self, key_id: str) -> None:
        ttl = self._settings.clerk_jwks_negative_kid_cache_seconds
        if ttl <= 0:
            return
        self._prune_negative_kid_cache()
        self._negative_kid_cache[_kid_cache_key(key_id)] = monotonic() + ttl

    def _prune_negative_kid_cache(self) -> None:
        now = monotonic()
        expired_keys = [
            cache_key
            for cache_key, expires_at in self._negative_kid_cache.items()
            if expires_at <= now
        ]
        for cache_key in expired_keys:
            self._negative_kid_cache.pop(cache_key, None)


def get_jwks_cache(app_settings: Settings = settings) -> ClerkJwksCache:
    global _jwks_cache
    if _jwks_cache is None:
        _jwks_cache = ClerkJwksCache(app_settings)
    return _jwks_cache


def reset_jwks_cache() -> None:
    global _jwks_cache
    _jwks_cache = None


async def get_user_repository() -> UserRepository:
    return MongoUserRepository()


async def get_optional_principal(
    request: Request = None,
    authorization: str | None = Header(default=None),
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthPrincipal | None:
    if authorization is None or not authorization.strip():
        return None
    return await _authenticate_authorization_header(
        authorization,
        user_repository,
        _settings_from_request(request),
    )


async def require_authenticated_principal(
    request: Request = None,
    authorization: str | None = Header(default=None, include_in_schema=False),
    _credentials: HTTPAuthorizationCredentials | None = Security(clerk_bearer_scheme),
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthPrincipal:
    app_settings = _settings_from_request(request)
    if dev_auth_bypass_enabled(app_settings):
        return development_auth_principal()
    if authorization is None or not authorization.strip():
        raise AuthenticationError(
            "Authentication is required.", category="missing_authorization_header"
        )
    return await _authenticate_authorization_header(
        authorization,
        user_repository,
        app_settings,
    )


async def require_clerk_user(
    principal: AuthPrincipal = Depends(require_authenticated_principal),
) -> AuthPrincipal:
    if principal.principal_type != "clerk_user" or not principal.user_id:
        raise AuthenticationError(
            "A Clerk user is required.", category="not_a_clerk_user"
        )
    return principal


async def verify_clerk_jwt(
    token: str,
    *,
    app_settings: Settings = settings,
) -> dict[str, Any]:
    header = _decode_json_segment(token, segment_index=0)
    key_id = header.get("kid")
    algorithm = header.get("alg")
    if not isinstance(key_id, str) or not key_id.strip():
        raise AuthenticationError(
            "Token signing key is missing.", category="missing_key_id"
        )
    if not isinstance(algorithm, str) or algorithm.lower() == "none":
        raise AuthenticationError(
            "Token algorithm is invalid.", category="invalid_algorithm"
        )

    key = await get_jwks_cache(app_settings).get_key(key_id)
    if key.get("alg") and key.get("alg") != algorithm:
        raise AuthenticationError(
            "Token signing algorithm is invalid.", category="invalid_algorithm"
        )

    claims = _decode_and_verify_signature(token, key, algorithm)
    _validate_clerk_claims(claims, app_settings)
    return claims


def dev_auth_bypass_enabled(app_settings: Settings) -> bool:
    """Fail-closed local auth bypass gate for Swagger/manual development only."""

    return app_settings.app_env == "development" and app_settings.dev_auth_bypass


def development_auth_principal() -> AuthPrincipal:
    return AuthPrincipal(
        subject="dev-local-user",
        principal_type="clerk_user",
        user_id="dev-local-user",
        organisation_id=None,
        api_key_id=None,
        scopes=[],
    )


async def _authenticate_authorization_header(
    authorization: str,
    user_repository: UserRepository,
    app_settings: Settings,
) -> AuthPrincipal:
    await clerk_auth_rate_limit_hook(event_type="attempt")
    try:
        token = _extract_bearer_token(authorization)
        claims = await verify_clerk_jwt(token, app_settings=app_settings)
        principal = principal_from_clerk_claims(claims)
        await _upsert_application_user(claims, user_repository)
    except Exception:
        await clerk_auth_rate_limit_hook(event_type="failure")
        raise
    await clerk_auth_rate_limit_hook(event_type="success")
    return principal


async def clerk_auth_rate_limit_hook(*, event_type: str) -> None:
    """Hook point for future Clerk authentication rate limiting.

    The default implementation is intentionally a no-op. A later limiter can
    wrap or monkeypatch this hook without needing raw JWTs, JWKS contents, or
    signing key IDs.
    """

    return None


async def _upsert_application_user(
    claims: dict[str, Any],
    user_repository: UserRepository,
) -> None:
    profile = user_profile_from_clerk_claims(claims)
    await user_repository.upsert_user(profile)


def _extract_bearer_token(authorization: str) -> str:
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise AuthenticationError(
            "Bearer token is malformed.", category="invalid_bearer_scheme"
        )
    return parts[1]


def _decode_and_verify_signature(
    token: str,
    key: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    if key.get("kty") == "oct":
        return _decode_hmac_token(token, key, algorithm)
    return _decode_asymmetric_token(token, key, algorithm)


def _decode_asymmetric_token(
    token: str,
    key: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    try:
        import jwt
        from jwt.algorithms import RSAAlgorithm
    except ImportError as error:
        raise AuthenticationError(
            "JWT verification support is unavailable.",
            category="jwt_library_unavailable",
        ) from error

    try:
        signing_key = RSAAlgorithm.from_jwk(json.dumps(key))
        return jwt.decode(
            token,
            key=signing_key,
            algorithms=[algorithm],
            options={
                "require": ["exp"],
                "verify_aud": False,
                "verify_iss": False,
            },
        )
    except jwt.ExpiredSignatureError as error:
        raise AuthenticationError("Token is expired.", category="expired_token") from error
    except Exception as error:
        raise AuthenticationError(
            "Token signature or claims are invalid.", category="invalid_signature"
        ) from error


def _decode_hmac_token(
    token: str,
    key: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    digest = SUPPORTED_HMAC_ALGORITHMS.get(algorithm)
    if digest is None:
        raise AuthenticationError(
            "Token signing algorithm is invalid.", category="invalid_algorithm"
        )

    parts = token.split(".")
    if len(parts) != 3:
        raise AuthenticationError("Token format is invalid.", category="invalid_token_format")

    secret = _base64url_decode(str(key.get("k", "")))
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected_signature = hmac.new(secret, signing_input, digest).digest()
    supplied_signature = _base64url_decode(parts[2])
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise AuthenticationError(
            "Token signature is invalid.", category="invalid_signature"
        )

    claims = _decode_json_segment(token, segment_index=1)
    if not isinstance(claims, dict):
        raise AuthenticationError("Token claims are invalid.", category="invalid_claims")
    return claims


def _validate_clerk_claims(claims: dict[str, Any], app_settings: Settings) -> None:
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise AuthenticationError("Token subject is invalid.", category="missing_subject")

    issuer = claims.get("iss")
    if app_settings.clerk_issuer and issuer != app_settings.clerk_issuer:
        raise AuthenticationError("Token issuer is invalid.", category="invalid_issuer")

    now = datetime.now(UTC).timestamp()
    expires_at = _number_claim(claims.get("exp"))
    if expires_at is None or expires_at <= now:
        raise AuthenticationError("Token is expired.", category="expired_token")

    not_before = _number_claim(claims.get("nbf"))
    if not_before is not None and not_before > now:
        raise AuthenticationError(
            "Token is not active yet.", category="token_not_yet_valid"
        )

    audiences = app_settings.clerk_audience_list
    if audiences and not _claim_matches_any(claims.get("aud"), audiences):
        raise AuthenticationError("Token audience is invalid.", category="invalid_audience")

    authorized_parties = app_settings.clerk_authorized_party_list
    authorized_party = claims.get("azp") or claims.get("authorized_party")
    if authorized_parties and authorized_party not in authorized_parties:
        raise AuthenticationError(
            "Token authorized party is invalid.", category="invalid_authorized_party"
        )


def _decode_json_segment(token: str, *, segment_index: int) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthenticationError("Token format is invalid.", category="invalid_token_format")
    try:
        payload = json.loads(_base64url_decode(parts[segment_index]))
    except (ValueError, UnicodeDecodeError) as error:
        raise AuthenticationError(
            "Token format is invalid.", category="invalid_token_format"
        ) from error
    if not isinstance(payload, dict):
        raise AuthenticationError("Token format is invalid.", category="invalid_token_format")
    return payload


def _base64url_decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))
    except Exception as error:
        raise AuthenticationError(
            "Token format is invalid.", category="invalid_token_format"
        ) from error


def _find_jwk(jwks: dict[str, Any], key_id: str) -> dict[str, Any] | None:
    for key in jwks.get("keys", []):
        if isinstance(key, dict) and key.get("kid") == key_id:
            return key
    return None


def _settings_from_request(request: Request | None) -> Settings:
    if request is None:
        return settings
    return getattr(request.app.state, "settings", settings)


def _kid_cache_key(key_id: str) -> str:
    return hashlib.sha256(key_id.encode("utf-8", errors="ignore")).hexdigest()


def _number_claim(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _claim_matches_any(value: Any, allowed_values: list[str]) -> bool:
    if isinstance(value, str):
        return value in allowed_values
    if isinstance(value, list):
        return any(item in allowed_values for item in value if isinstance(item, str))
    return False

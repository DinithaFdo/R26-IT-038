from datetime import UTC, datetime
from typing import Any

from app.auth.schemas import AuthPrincipal


def principal_from_clerk_claims(
    claims: dict[str, Any],
    *,
    scopes: list[str] | None = None,
) -> AuthPrincipal:
    clerk_user_id = str(claims["sub"])
    return AuthPrincipal(
        subject=clerk_user_id,
        principal_type="clerk_user",
        user_id=clerk_user_id,
        organisation_id=_optional_string(
            claims.get("org_id") or claims.get("organization_id")
        ),
        api_key_id=None,
        scopes=scopes or _scope_list(claims),
    )


def user_profile_from_clerk_claims(claims: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC)
    profile = {
        "clerk_user_id": str(claims["sub"]),
        "updated_at": now,
        "last_seen_at": now,
    }

    email = _safe_claim_string(claims.get("email"))
    if email is not None:
        profile["email"] = email

    display_name = _safe_claim_string(
        claims.get("name")
        or claims.get("full_name")
        or claims.get("username")
        or claims.get("display_name")
    )
    if display_name is not None:
        profile["display_name"] = display_name

    return profile


def _scope_list(claims: dict[str, Any]) -> list[str]:
    raw_scope = claims.get("scope") or claims.get("scopes")
    if isinstance(raw_scope, str):
        return [scope for scope in raw_scope.split() if scope]
    if isinstance(raw_scope, list):
        return [
            str(scope)
            for scope in raw_scope
            if isinstance(scope, str) and scope.strip()
        ]
    return []


def _safe_claim_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped[:256] if stripped else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None

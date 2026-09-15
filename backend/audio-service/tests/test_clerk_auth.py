import asyncio
import base64
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

from app.auth import clerk_auth
from app.auth.clerk_auth import get_optional_principal
from app.config.settings import Settings
from app.config.settings import settings
from app.main import app

SECRET = b"test-secret"
KEY_ID = "test-key"
ISSUER = "https://example.clerk.accounts.dev"
AUDIENCE = "multi-scope"
AUTHORIZED_PARTY = "http://localhost:3000"


@pytest.fixture(autouse=True)
def configure_clerk(monkeypatch):
    monkeypatch.setattr(settings, "clerk_issuer", ISSUER)
    monkeypatch.setattr(settings, "clerk_jwks_url", "https://clerk.test/jwks.json")
    monkeypatch.setattr(settings, "clerk_audience", AUDIENCE)
    monkeypatch.setattr(settings, "clerk_authorized_parties", AUTHORIZED_PARTY)
    monkeypatch.setattr(settings, "clerk_jwks_cache_seconds", 3600)
    monkeypatch.setattr(clerk_auth, "_jwks_cache", FakeJwksCache([jwk()]))
    app.dependency_overrides[clerk_auth.get_user_repository] = lambda: FakeUserRepository()
    yield
    app.dependency_overrides.clear()
    clerk_auth.reset_jwks_cache()


def test_missing_token_returns_safe_401(client: TestClient) -> None:
    response = client.get("/api/v1/users/me", headers={"X-Request-ID": "auth-1"})

    assert_auth_error(response, "auth-1")


def test_malformed_token_returns_safe_401(client: TestClient) -> None:
    response = client.get(
        "/api/v1/users/me",
        headers={
            "Authorization": "not-a-bearer-token",
            "X-Request-ID": "auth-2",
        },
    )

    assert_auth_error(response, "auth-2")


def test_expired_token_returns_safe_401(client: TestClient) -> None:
    response = authenticated_get(
        client,
        make_token({"exp": timestamp(-10)}),
        request_id="auth-expired",
    )

    assert_auth_error(response, "auth-expired")


def test_not_before_token_returns_safe_401(client: TestClient) -> None:
    response = authenticated_get(
        client,
        make_token({"nbf": timestamp(60)}),
        request_id="auth-nbf",
    )

    assert_auth_error(response, "auth-nbf")


def test_invalid_signature_returns_safe_401(client: TestClient) -> None:
    token = make_token()
    header, claims, signature = token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    tampered_token = f"{header}.{claims}.{replacement}{signature[1:]}"

    response = authenticated_get(client, tampered_token, request_id="auth-signature")

    assert_auth_error(response, "auth-signature")


def test_incorrect_issuer_returns_safe_401(client: TestClient) -> None:
    response = authenticated_get(
        client,
        make_token({"iss": "https://wrong-issuer.example"}),
        request_id="auth-issuer",
    )

    assert_auth_error(response, "auth-issuer")


def test_incorrect_audience_returns_safe_401(client: TestClient) -> None:
    response = authenticated_get(
        client,
        make_token({"aud": "wrong-audience"}),
        request_id="auth-audience",
    )

    assert_auth_error(response, "auth-audience")


def test_incorrect_authorized_party_returns_safe_401(client: TestClient) -> None:
    response = authenticated_get(
        client,
        make_token({"azp": "https://evil.example"}),
        request_id="auth-azp",
    )

    assert_auth_error(response, "auth-azp")


def test_missing_subject_returns_safe_401(client: TestClient) -> None:
    response = authenticated_get(
        client,
        make_token({"sub": ""}),
        request_id="auth-subject",
    )

    assert_auth_error(response, "auth-subject")


# --- Regression coverage for the 2026-08-09 "every request 401s" incident ---
#
# Root cause: CLERK_ISSUER / CLERK_JWKS_URL were blank in backend/.env. The
# JWKS fetch is refused before any token is even inspected, so every request
# fails identically regardless of whether the user is genuinely signed in.
# `app_env=development` meant the production-only settings validator that
# would normally require these never ran, so the misconfiguration was silent
# at startup. These tests pin (a) the client-visible behaviour, which must stay
# the generic 401 body, and (b) the internal failure category, which must stay
# distinguishable in logs so this specific failure mode is fast to diagnose.


def test_unconfigured_jwks_url_returns_the_same_safe_401_as_any_other_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "clerk_jwks_url", "")
    clerk_auth.reset_jwks_cache()

    response = authenticated_get(client, make_token(), request_id="auth-no-jwks")

    # The client must never learn *why* auth failed -- a misconfigured server
    # and a bad token are indistinguishable from the outside.
    assert_auth_error(response, "auth-no-jwks")


@pytest.mark.anyio
async def test_unconfigured_jwks_url_raises_a_distinguishable_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "clerk_jwks_url", "")
    clerk_auth.reset_jwks_cache()

    with pytest.raises(clerk_auth.AuthenticationError) as excinfo:
        await clerk_auth.verify_clerk_jwt(make_token(), app_settings=settings)

    assert excinfo.value.category == "missing_jwks_configuration"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("token_factory", "expected_category"),
    [
        (lambda: make_token({"exp": timestamp(-10)}), "expired_token"),
        (lambda: make_token({"iss": "https://wrong-issuer.example"}), "invalid_issuer"),
        (lambda: make_token({"aud": "wrong-audience"}), "invalid_audience"),
        (
            lambda: make_token({"azp": "https://evil.example"}),
            "invalid_authorized_party",
        ),
        (lambda: make_token({"sub": ""}), "missing_subject"),
        (lambda: make_token({"nbf": timestamp(60)}), "token_not_yet_valid"),
    ],
)
async def test_each_claim_failure_has_its_own_category(
    token_factory,
    expected_category: str,
) -> None:
    with pytest.raises(clerk_auth.AuthenticationError) as excinfo:
        await clerk_auth.verify_clerk_jwt(token_factory(), app_settings=settings)

    assert excinfo.value.category == expected_category


@pytest.mark.anyio
async def test_invalid_signature_has_its_own_category() -> None:
    token = make_token()
    header, claims, signature = token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    tampered_token = f"{header}.{claims}.{replacement}{signature[1:]}"

    with pytest.raises(clerk_auth.AuthenticationError) as excinfo:
        await clerk_auth.verify_clerk_jwt(tampered_token, app_settings=settings)

    assert excinfo.value.category == "invalid_signature"


def test_missing_authorization_header_has_its_own_category() -> None:
    with pytest.raises(clerk_auth.AuthenticationError) as excinfo:
        asyncio.run(
            clerk_auth.require_authenticated_principal(
                authorization=None,
                user_repository=FakeUserRepository(),
            )
        )

    assert excinfo.value.category == "missing_authorization_header"


def test_development_with_dev_auth_bypass_false_requires_auth(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "dev_auth_bypass", False)

    response = client.get("/api/v1/users/me", headers={"X-Request-ID": "auth-dev-off"})

    assert_auth_error(response, "auth-dev-off")


def test_development_with_dev_auth_bypass_true_returns_synthetic_principal(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "dev_auth_bypass", True)

    response = client.get("/api/v1/users/me")

    assert response.status_code == 200
    assert response.json() == {
        "subject": "dev-local-user",
        "principal_type": "clerk_user",
        "user_id": "dev-local-user",
        "organisation_id": None,
        "api_key_id": None,
        "scopes": [],
    }


@pytest.mark.parametrize("app_env", ["production", "staging", "test", "", "unknown"])
def test_dev_auth_bypass_fails_closed_outside_development(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    app_env: str,
) -> None:
    monkeypatch.setattr(settings, "app_env", app_env)
    monkeypatch.setattr(settings, "dev_auth_bypass", True)

    response = client.get("/api/v1/users/me", headers={"X-Request-ID": "auth-closed"})

    assert_auth_error(response, "auth-closed")


def test_dev_auth_bypass_does_not_change_valid_production_auth_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "dev_auth_bypass", True)

    response = authenticated_get(client, make_token(), request_id="auth-prod-real")

    assert response.status_code == 200
    assert response.json()["user_id"] == "user_123"


def test_dev_auth_bypass_enabled_helper_is_exact_and_fail_closed() -> None:
    assert clerk_auth.dev_auth_bypass_enabled(
        make_clerk_settings(app_env="development", dev_auth_bypass=True)
    )
    assert not clerk_auth.dev_auth_bypass_enabled(
        make_clerk_settings(app_env="testing", dev_auth_bypass=True)
    )
    assert not clerk_auth.dev_auth_bypass_enabled(
        SimpleNamespace(app_env="production", dev_auth_bypass=True)
    )


def test_development_principal_has_no_elevated_privileges() -> None:
    principal = clerk_auth.development_auth_principal()

    assert principal.principal_type == "clerk_user"
    assert principal.user_id == "dev-local-user"
    assert principal.organisation_id is None
    assert principal.api_key_id is None
    assert principal.scopes == []
    assert "admin" not in principal.model_dump_json()


def test_authentication_error_category_defaults_to_unspecified() -> None:
    """A call site that forgets to pass a category must still work."""

    from app.core.exceptions import AuthenticationError

    assert AuthenticationError("generic failure").category == "unspecified"


def test_valid_token_returns_sanitized_principal_and_upserts_user(
    client: TestClient,
) -> None:
    repository = FakeUserRepository()
    app.dependency_overrides[clerk_auth.get_user_repository] = lambda: repository
    token = make_token(
        {
            "sub": "user_123",
            "email": "researcher@example.edu",
            "name": "Researcher One",
        }
    )

    response = authenticated_get(client, token, request_id="auth-valid")

    assert response.status_code == 200
    assert response.json() == {
        "subject": "user_123",
        "principal_type": "clerk_user",
        "user_id": "user_123",
        "organisation_id": None,
        "api_key_id": None,
        "scopes": [],
    }
    assert repository.saved_user["clerk_user_id"] == "user_123"
    assert repository.saved_user["email"] == "researcher@example.edu"
    assert repository.saved_user["display_name"] == "Researcher One"
    assert repository.saved_user["last_seen_at"].tzinfo is not None
    assert "claims" not in response.text


@pytest.mark.anyio
async def test_known_key_uses_jwks_cache(monkeypatch) -> None:
    cache = ControlledJwksCache(
        make_clerk_settings(),
        payloads=[{"keys": [jwk()]}],
    )
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    first = await clerk_auth.get_jwks_cache().get_key(KEY_ID)
    second = await clerk_auth.get_jwks_cache().get_key(KEY_ID)

    assert first == second
    assert cache.refresh_count == 1


@pytest.mark.anyio
async def test_first_unknown_key_id_refreshes_once(monkeypatch) -> None:
    cache = ControlledJwksCache(
        make_clerk_settings(),
        payloads=[{"keys": [jwk(key_id="old-key")]}],
    )
    cache.prime({"keys": [jwk(key_id="old-key")]})
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    with pytest.raises(clerk_auth.AuthenticationError):
        await clerk_auth.get_jwks_cache().get_key("missing-key")

    assert cache.refresh_count == 1


@pytest.mark.anyio
async def test_repeated_unknown_key_id_uses_negative_cache(monkeypatch) -> None:
    cache = ControlledJwksCache(
        make_clerk_settings(),
        payloads=[{"keys": [jwk(key_id="old-key")]}],
    )
    cache.prime({"keys": [jwk(key_id="old-key")]})
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    for _ in range(3):
        with pytest.raises(clerk_auth.AuthenticationError):
            await clerk_auth.get_jwks_cache().get_key("missing-key")

    assert cache.refresh_count == 1


@pytest.mark.anyio
async def test_concurrent_unknown_key_ids_use_single_flight_refresh(
    monkeypatch,
) -> None:
    cache = ControlledJwksCache(
        make_clerk_settings(),
        payloads=[{"keys": [jwk(key_id="old-key")]}],
        fetch_delay_seconds=0.01,
    )
    cache.prime({"keys": [jwk(key_id="old-key")]})
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    async def lookup_missing_key() -> None:
        with pytest.raises(clerk_auth.AuthenticationError):
            await clerk_auth.get_jwks_cache().get_key("missing-key")

    await asyncio.gather(*(lookup_missing_key() for _ in range(10)))

    assert cache.refresh_count == 1


@pytest.mark.anyio
async def test_negative_key_id_cache_expires(monkeypatch) -> None:
    now = 1000.0

    def fake_monotonic() -> float:
        return now

    monkeypatch.setattr(clerk_auth, "monotonic", fake_monotonic)
    cache = ControlledJwksCache(
        make_clerk_settings(
            clerk_jwks_min_refresh_interval_seconds=0,
            clerk_jwks_negative_kid_cache_seconds=1,
        ),
        payloads=[
            {"keys": [jwk(key_id="old-key")]},
            {"keys": [jwk(key_id="old-key")]},
        ],
    )
    cache.prime({"keys": [jwk(key_id="old-key")]})
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    with pytest.raises(clerk_auth.AuthenticationError):
        await clerk_auth.get_jwks_cache().get_key("missing-key")
    with pytest.raises(clerk_auth.AuthenticationError):
        await clerk_auth.get_jwks_cache().get_key("missing-key")

    now += 2
    with pytest.raises(clerk_auth.AuthenticationError):
        await clerk_auth.get_jwks_cache().get_key("missing-key")

    assert cache.refresh_count == 2


@pytest.mark.anyio
async def test_valid_rotated_rsa_key_works_after_refresh(monkeypatch) -> None:
    jwt = pytest.importorskip("jwt")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")

    old_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rotated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = make_rsa_token(jwt, rotated_key, key_id="rotated-key")
    cache = ControlledJwksCache(
        make_clerk_settings(),
        payloads=[
            {"keys": [rsa_jwk(old_key, "old-key"), rsa_jwk(rotated_key, "rotated-key")]},
        ],
    )
    cache.prime({"keys": [rsa_jwk(old_key, "old-key")]})
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    claims = await clerk_auth.verify_clerk_jwt(token)

    assert claims["sub"] == "user_123"
    assert cache.refresh_count == 1
    assert serialization is not None


def test_clerk_network_failure_returns_sanitized_current_auth_contract(
    client: TestClient,
    monkeypatch,
) -> None:
    cache = ControlledJwksCache(
        make_clerk_settings(),
        payloads=[RuntimeError("internal network failure for jwks endpoint")],
    )
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    response = authenticated_get(client, make_token(), request_id="auth-network")

    assert_auth_error(response, "auth-network")
    assert "internal network failure" not in response.text
    assert "jwks" not in response.text.lower()


@pytest.mark.anyio
async def test_clerk_auth_rate_limit_hook_is_called(monkeypatch) -> None:
    events = []

    async def hook(*, event_type: str) -> None:
        events.append(event_type)

    monkeypatch.setattr(clerk_auth, "clerk_auth_rate_limit_hook", hook)

    await clerk_auth.get_optional_principal(
        authorization=f"Bearer {make_token()}",
        user_repository=FakeUserRepository(),
    )
    with pytest.raises(clerk_auth.AuthenticationError):
        await clerk_auth.get_optional_principal(
            authorization="Bearer malformed",
            user_repository=FakeUserRepository(),
        )

    assert events == ["attempt", "success", "attempt", "failure"]


@pytest.mark.anyio
async def test_optional_principal_allows_missing_token() -> None:
    principal = await get_optional_principal(
        authorization=None,
        user_repository=FakeUserRepository(),
    )

    assert principal is None


@pytest.mark.anyio
async def test_jwks_cache_refreshes_when_key_id_is_unknown(monkeypatch) -> None:
    cache = FakeJwksCache([jwk(key_id="old-key")], refreshed_keys=[jwk()])
    monkeypatch.setattr(clerk_auth, "_jwks_cache", cache)

    claims = await clerk_auth.verify_clerk_jwt(make_token())

    assert claims["sub"] == "user_123"
    assert cache.refresh_count == 1


def authenticated_get(client: TestClient, token: str, *, request_id: str):
    return client.get(
        "/api/v1/users/me",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Request-ID": request_id,
        },
    )


def assert_auth_error(response, request_id: str) -> None:
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    server_request_id = response.headers["X-Request-ID"]
    UUID(server_request_id)
    assert server_request_id != request_id
    assert response.headers["X-Client-Correlation-ID"] == request_id
    assert response.json() == {
        "request_id": server_request_id,
        "error": {
            "code": "authentication_failed",
            "message": "Authentication failed.",
            "details": None,
        },
    }
    assert "Traceback" not in response.text
    assert "test-secret" not in response.text
    assert "eyJ" not in response.text


def make_token(overrides: dict[str, Any] | None = None, *, key_id: str = KEY_ID) -> str:
    header = {
        "alg": "HS256",
        "typ": "JWT",
        "kid": key_id,
    }
    claims = {
        "iss": ISSUER,
        "sub": "user_123",
        "aud": AUDIENCE,
        "azp": AUTHORIZED_PARTY,
        "exp": timestamp(300),
        "iat": timestamp(-10),
    }
    if overrides:
        claims.update(overrides)

    signing_input = (
        f"{b64json(header).decode('ascii')}.{b64json(claims).decode('ascii')}"
    )
    signature = hmac.new(
        SECRET,
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{b64(signature).decode('ascii')}"


def jwk(*, key_id: str = KEY_ID) -> dict[str, str]:
    return {
        "kty": "oct",
        "kid": key_id,
        "alg": "HS256",
        "k": b64(SECRET).decode("ascii"),
    }


def b64json(payload: dict[str, Any]) -> bytes:
    return b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def b64(payload: bytes) -> bytes:
    return base64.urlsafe_b64encode(payload).rstrip(b"=")


def timestamp(offset_seconds: int) -> int:
    return int((datetime.now(UTC) + timedelta(seconds=offset_seconds)).timestamp())


class FakeJwksCache:
    def __init__(
        self,
        keys: list[dict[str, Any]],
        *,
        refreshed_keys: list[dict[str, Any]] | None = None,
    ) -> None:
        self.keys = keys
        self.refreshed_keys = refreshed_keys or keys
        self.refresh_count = 0

    async def get_key(self, key_id: str) -> dict[str, Any]:
        for key in self.keys:
            if key["kid"] == key_id:
                return key
        self.refresh_count += 1
        self.keys = self.refreshed_keys
        for key in self.keys:
            if key["kid"] == key_id:
                return key
        raise AssertionError("Unknown test key.")


class ControlledJwksCache(clerk_auth.ClerkJwksCache):
    def __init__(
        self,
        app_settings: Settings,
        *,
        payloads: list[dict[str, Any] | Exception],
        fetch_delay_seconds: float = 0.0,
    ) -> None:
        super().__init__(app_settings)
        self.payloads = payloads
        self.fetch_delay_seconds = fetch_delay_seconds
        self.refresh_count = 0

    def prime(self, jwks: dict[str, Any]) -> None:
        self._jwks = jwks
        self._expires_at = (
            clerk_auth.monotonic() + self._settings.clerk_jwks_cache_seconds
        )

    async def _fetch_jwks(self) -> dict[str, Any]:
        self.refresh_count += 1
        if self.fetch_delay_seconds:
            await asyncio.sleep(self.fetch_delay_seconds)
        payload_index = min(self.refresh_count - 1, len(self.payloads) - 1)
        payload = self.payloads[payload_index]
        if isinstance(payload, Exception):
            raise payload
        return payload


class FakeUserRepository:
    def __init__(self) -> None:
        self.saved_user = None

    async def get_user(self, user_id: str):
        return None

    async def get_user_by_clerk_user_id(self, clerk_user_id: str):
        return None

    async def upsert_user(self, user: dict[str, Any]):
        self.saved_user = user
        return user


def make_clerk_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "clerk_issuer": ISSUER,
        "clerk_jwks_url": "https://clerk.test/jwks.json",
        "clerk_audience": AUDIENCE,
        "clerk_authorized_parties": AUTHORIZED_PARTY,
        "clerk_jwks_cache_seconds": 3600,
        "clerk_jwks_min_refresh_interval_seconds": 60,
        "clerk_jwks_negative_kid_cache_seconds": 60,
    }
    values.update(overrides)
    return Settings(**values)


def make_rsa_token(jwt, private_key, *, key_id: str) -> str:
    return jwt.encode(
        {
            "iss": ISSUER,
            "sub": "user_123",
            "aud": AUDIENCE,
            "azp": AUTHORIZED_PARTY,
            "exp": timestamp(300),
            "iat": timestamp(-10),
        },
        key=private_key,
        algorithm="RS256",
        headers={"kid": key_id, "typ": "JWT"},
    )


def rsa_jwk(private_key, key_id: str) -> dict[str, str]:
    public_numbers = private_key.public_key().public_numbers()
    return {
        "kty": "RSA",
        "kid": key_id,
        "alg": "RS256",
        "use": "sig",
        "n": b64(_int_to_bytes(public_numbers.n)).decode("ascii"),
        "e": b64(_int_to_bytes(public_numbers.e)).decode("ascii"),
    }


def _int_to_bytes(value: int) -> bytes:
    return value.to_bytes((value.bit_length() + 7) // 8, "big")

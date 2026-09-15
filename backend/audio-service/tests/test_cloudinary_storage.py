from datetime import datetime
import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.schemas.common import BranchStatus
from app.storage.cloudinary_storage import CloudinaryAudioStorage
from app.storage.factory import get_audio_storage, storage_readiness
from app.storage.noop import NoOpAudioStorage


async def wait_until(predicate, *, timeout_seconds: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate()


def make_cloudinary_settings(tmp_path, **overrides) -> Settings:
    values = {
        "upload_dir": str(tmp_path / "uploads"),
        "cloudinary_cloud_name": "demo-cloud",
        "cloudinary_api_key": "demo-key",
        "cloudinary_api_secret": "demo-secret",
        "cloudinary_audio_folder": "multiscope/audio",
        "cloudinary_storage_enabled": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.anyio
async def test_noop_storage_is_disabled(tmp_path) -> None:
    storage = NoOpAudioStorage()

    metadata = await storage.upload_audio(
        tmp_path / "sample.wav",
        owner_user_id="user_123",
        audio_id="audio_123",
    )

    assert metadata.status == BranchStatus.skipped
    assert await storage.health() == {
        "storage_enabled": False,
        "storage_available": False,
    }


@pytest.mark.anyio
async def test_storage_factory_returns_noop_when_disabled(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path, cloudinary_storage_enabled=False)

    assert isinstance(get_audio_storage(settings), NoOpAudioStorage)
    assert await storage_readiness(settings) == {
        "storage_enabled": False,
        "storage_available": False,
    }


@pytest.mark.anyio
async def test_storage_factory_returns_noop_when_policy_disabled_in_development(
    tmp_path,
) -> None:
    settings = make_cloudinary_settings(tmp_path, storage_policy="disabled")

    assert isinstance(get_audio_storage(settings), NoOpAudioStorage)
    assert await storage_readiness(settings) == {
        "storage_enabled": False,
        "storage_available": False,
    }


@pytest.mark.anyio
async def test_storage_factory_optional_mode_is_noop_without_cloudinary(
    tmp_path,
) -> None:
    settings = make_cloudinary_settings(
        tmp_path,
        cloudinary_storage_enabled=False,
        storage_policy="optional",
    )

    assert isinstance(get_audio_storage(settings), NoOpAudioStorage)
    assert await storage_readiness(settings) == {
        "storage_enabled": False,
        "storage_available": False,
    }


@pytest.mark.anyio
async def test_cloudinary_health_reports_missing_credentials(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path, cloudinary_api_secret="")
    storage = CloudinaryAudioStorage(settings)

    assert await storage.health() == {
        "storage_enabled": True,
        "storage_available": False,
        "storage_reconciliation_backlog": 0,
    }


@pytest.mark.anyio
async def test_cloudinary_upload_uses_private_video_resource(tmp_path) -> None:
    settings = make_cloudinary_settings(
        tmp_path,
        cloudinary_sdk_timeout_seconds=12,
        cloudinary_upload_timeout_seconds=30,
    )
    storage = CloudinaryAudioStorage(settings)
    recorder = FakeCloudinaryModules()
    storage._load_cloudinary_modules = lambda: recorder.modules()
    audio_path = tmp_path / "validated.wav"
    audio_path.write_bytes(b"audio")

    metadata = await storage.upload_audio(
        audio_path,
        owner_user_id="user_123",
        audio_id="audio_123",
        content_type="audio/wav",
    )

    assert metadata.status == BranchStatus.success
    assert metadata.asset_id == "asset-123"
    assert metadata.public_id == "multiscope/audio/user_123/audio_123"
    assert metadata.resource_type == "video"
    assert metadata.bytes == 1024
    assert isinstance(metadata.created_at, datetime)
    assert "secure_url" not in metadata.model_dump(mode="json")
    assert recorder.upload_kwargs["resource_type"] == "video"
    assert recorder.upload_kwargs["type"] == "authenticated"
    assert recorder.upload_kwargs["public_id"] == "multiscope/audio/user_123/audio_123"
    assert recorder.upload_kwargs["overwrite"] is False
    assert recorder.upload_kwargs["timeout"] == 12


@pytest.mark.anyio
async def test_cloudinary_upload_failure_returns_safe_metadata(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path)
    storage = CloudinaryAudioStorage(settings)
    recorder = FakeCloudinaryModules(raise_upload=True)
    storage._load_cloudinary_modules = lambda: recorder.modules()
    audio_path = tmp_path / "validated.wav"
    audio_path.write_bytes(b"audio")

    metadata = await storage.upload_audio(
        audio_path,
        owner_user_id="user_123",
        audio_id="audio_123",
    )

    assert metadata.status == BranchStatus.failed
    assert metadata.error == "Audio storage failed."
    assert metadata.asset_id is None
    assert recorder.resource_calls == 0


@pytest.mark.anyio
async def test_cloudinary_outer_timeout_then_no_asset_exists(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path, cloudinary_upload_timeout_seconds=0.01)
    storage = CloudinaryAudioStorage(settings)
    recorder = FakeCloudinaryModules(upload_sleep_seconds=0.05)
    storage._load_cloudinary_modules = lambda: recorder.modules()
    audio_path = tmp_path / "validated.wav"
    audio_path.write_bytes(b"audio")

    metadata = await storage.upload_audio(
        audio_path,
        owner_user_id="user_123",
        audio_id="audio_123",
    )

    assert metadata.status == BranchStatus.failed
    assert metadata.error == "Audio storage timed out; outcome is being reconciled."
    await wait_until(lambda: recorder.resource_calls == 1)
    assert recorder.resource_calls == 1
    assert recorder.destroyed_public_ids == []
    assert audio_path.exists()


@pytest.mark.anyio
async def test_cloudinary_outer_timeout_then_late_asset_is_deleted(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path, cloudinary_upload_timeout_seconds=0.01)
    recorded_events = []

    async def record_event(event):
        recorded_events.append(event)

    storage = CloudinaryAudioStorage(settings, reconciliation_recorder=record_event)
    recorder = FakeCloudinaryModules(
        upload_sleep_seconds=0.05,
        resource_results=["exists"],
    )
    storage._load_cloudinary_modules = lambda: recorder.modules()
    audio_path = tmp_path / "validated.wav"
    audio_path.write_bytes(b"audio")

    metadata = await storage.upload_audio(
        audio_path,
        owner_user_id="user_123",
        audio_id="audio_123",
    )
    await wait_until(
        lambda: recorder.destroyed_public_ids
        == ["multiscope/audio/user_123/audio_123"]
        and bool(recorded_events)
    )

    assert metadata.status == BranchStatus.failed
    assert recorder.destroyed_public_ids == ["multiscope/audio/user_123/audio_123"]
    assert recorded_events[-1].event_type == (
        "cloudinary_reconciliation_orphan_deleted"
    )
    assert recorded_events[-1].request_id == "audio_123"


@pytest.mark.anyio
async def test_cloudinary_reconciliation_deletion_fails_safely(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path)
    storage = CloudinaryAudioStorage(settings)
    recorder = FakeCloudinaryModules(resource_results=["exists"], raise_destroy=True)
    storage._load_cloudinary_modules = lambda: recorder.modules()

    event = await storage.reconcile_upload(
        public_id="multiscope/audio/user_123/audio_123",
        owner_user_id="user_123",
    )

    assert event.event_type == "cloudinary_reconciliation_delete_failed"
    assert event.asset_exists is True
    assert event.deleted is False
    assert event.deletion_failed is True


@pytest.mark.anyio
async def test_cloudinary_reconciliation_preserves_owned_asset(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path)
    storage = CloudinaryAudioStorage(settings)
    recorder = FakeCloudinaryModules(resource_results=["exists"])
    storage._load_cloudinary_modules = lambda: recorder.modules()

    event = await storage.reconcile_upload(
        public_id="multiscope/audio/user_123/audio_123",
        owner_user_id="user_123",
        prediction_owns_asset=True,
    )

    assert event.event_type == "cloudinary_reconciliation_asset_owned"
    assert event.deleted is False
    assert recorder.destroyed_public_ids == []


@pytest.mark.anyio
async def test_cloudinary_logs_do_not_include_credentials(tmp_path, caplog) -> None:
    settings = make_cloudinary_settings(
        tmp_path,
        cloudinary_cloud_name="secret-cloud",
        cloudinary_api_key="secret-key",
        cloudinary_api_secret="secret-value",
    )
    storage = CloudinaryAudioStorage(settings)
    recorder = FakeCloudinaryModules(raise_upload=True)
    storage._load_cloudinary_modules = lambda: recorder.modules()
    audio_path = tmp_path / "validated.wav"
    audio_path.write_bytes(b"audio")

    await storage.upload_audio(
        audio_path,
        owner_user_id="user_123",
        audio_id="audio_123",
    )

    assert "secret-cloud" not in caplog.text
    assert "secret-key" not in caplog.text
    assert "secret-value" not in caplog.text


@pytest.mark.anyio
async def test_signed_playback_url_requires_owner_and_authenticated_type(
    tmp_path,
) -> None:
    settings = make_cloudinary_settings(tmp_path)
    storage = CloudinaryAudioStorage(settings)
    recorder = FakeCloudinaryModules()
    storage._load_cloudinary_modules = lambda: recorder.modules()

    url = await storage.generate_signed_playback_url(
        "multiscope/audio/user_123/audio_123",
        owner_user_id="user_123",
        expires_in_seconds=60,
    )

    assert url == "https://res.cloudinary.com/demo/video/authenticated/signed"
    assert recorder.url_kwargs["resource_type"] == "video"
    assert recorder.url_kwargs["type"] == "authenticated"
    assert recorder.url_kwargs["sign_url"] is True
    assert recorder.url_kwargs["secure"] is True
    assert "expires_at" in recorder.url_kwargs


@pytest.mark.anyio
async def test_signed_playback_url_rejects_non_owner(tmp_path) -> None:
    settings = make_cloudinary_settings(tmp_path)
    storage = CloudinaryAudioStorage(settings)

    with pytest.raises(PermissionError):
        await storage.generate_signed_playback_url(
            "multiscope/audio/user_123/audio_123",
            owner_user_id="user_456",
        )


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.cloudinary
@pytest.mark.network
async def test_real_cloudinary_health_is_skipped_without_credentials(tmp_path) -> None:
    required = [
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ]
    if not all(os.getenv(name) for name in required):
        pytest.skip("Cloudinary credentials are not configured.")

    try:
        __import__("cloudinary")
    except ImportError:
        pytest.skip("Cloudinary SDK is not installed in the test environment.")

    settings = make_cloudinary_settings(
        tmp_path,
        cloudinary_cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        cloudinary_api_key=os.environ["CLOUDINARY_API_KEY"],
        cloudinary_api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )

    health = await CloudinaryAudioStorage(settings).health()

    assert health == {"storage_enabled": True, "storage_available": True}


class FakeCloudinaryModules:
    def __init__(
        self,
        *,
        raise_upload: bool = False,
        raise_destroy: bool = False,
        upload_sleep_seconds: float = 0.0,
        resource_results: list[str | None] | None = None,
    ) -> None:
        self.raise_upload = raise_upload
        self.raise_destroy = raise_destroy
        self.upload_sleep_seconds = upload_sleep_seconds
        self.resource_results = resource_results or []
        self.resource_calls = 0
        self.upload_kwargs = None
        self.url_kwargs = None
        self.destroyed_public_ids = []
        self.cloudinary = SimpleNamespace(config=lambda **_kwargs: None)
        self.uploader = SimpleNamespace(
            upload=self.upload,
            destroy=self.destroy,
        )
        self.api = SimpleNamespace(resource=self.resource)
        self.utils = SimpleNamespace(cloudinary_url=self.cloudinary_url)

    def modules(self):
        return self.cloudinary, self.uploader, self.utils, self.api

    def upload(self, _source_path, **kwargs):
        self.upload_kwargs = kwargs
        if self.upload_sleep_seconds:
            time.sleep(self.upload_sleep_seconds)
        if self.raise_upload:
            raise RuntimeError("secret internal cloudinary failure")
        return {
            "asset_id": "asset-123",
            "public_id": kwargs["public_id"],
            "resource_type": "video",
            "version": 123,
            "format": "wav",
            "bytes": 1024,
            "duration": 1.5,
            "created_at": "2026-07-26T00:00:00Z",
            "secure_url": "https://public-url-that-must-not-be-stored",
        }

    def resource(self, public_id, **_kwargs):
        self.resource_calls += 1
        result = (
            self.resource_results.pop(0)
            if self.resource_results
            else None
        )
        if result is None:
            raise CloudinaryNotFound("not found")
        return {"public_id": public_id, "resource_type": "video"}

    def destroy(self, public_id, **_kwargs):
        if self.raise_destroy:
            raise RuntimeError("secret internal destroy failure")
        self.destroyed_public_ids.append(public_id)
        return {"result": "ok"}

    def cloudinary_url(self, _public_id, **kwargs):
        self.url_kwargs = kwargs
        return "https://res.cloudinary.com/demo/video/authenticated/signed", {}


class CloudinaryNotFound(Exception):
    status_code = 404

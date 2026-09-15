from app.config.settings import Settings, settings
from app.storage.cloudinary_storage import CloudinaryAudioStorage
from app.storage.noop import NoOpAudioStorage
from app.storage.protocols import AudioStorage


def get_audio_storage(app_settings: Settings = settings) -> AudioStorage:
    if app_settings.storage_policy == "disabled":
        if app_settings.app_env.lower() not in {"development", "local", "test", "testing"}:
            raise RuntimeError("Disabled audio storage is allowed only in local/test modes.")
        return NoOpAudioStorage(reason="Audio storage is disabled by policy.")
    if app_settings.cloudinary_storage_enabled:
        return CloudinaryAudioStorage(app_settings)
    if app_settings.storage_policy == "optional":
        return NoOpAudioStorage(reason="Audio storage is unavailable in optional mode.")
    if app_settings.app_env.lower() not in {"development", "local", "test", "testing"}:
        raise RuntimeError("No-op audio storage is allowed only in local/test modes.")
    return NoOpAudioStorage(reason="Audio storage is disabled.")


async def storage_readiness(app_settings: Settings = settings) -> dict:
    return await get_audio_storage(app_settings).health()

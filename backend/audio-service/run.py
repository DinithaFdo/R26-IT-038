from pathlib import Path

from runtime_guard import ensure_supported_python


def main() -> None:
    ensure_supported_python()

    import uvicorn

    from app.config.settings import settings

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        reload_dirs=[str(Path(__file__).resolve().parent / "app")] if settings.debug else None,
    )


if __name__ == "__main__":
    main()

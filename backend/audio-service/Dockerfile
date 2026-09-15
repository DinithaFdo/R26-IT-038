FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    DEBUG=false \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock pyproject.toml ./
# The CPU-only torch index avoids pulling the CUDA runtime, which would add
# several gigabytes to the image for no benefit in a CPU container.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -c requirements.lock ".[xai]" \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"

RUN groupadd --system multiscope \
    && useradd --system --gid multiscope --home-dir /app --shell /usr/sbin/nologin multiscope

COPY app ./app
COPY mcp_server ./mcp_server
COPY run.py runtime_guard.py ./
COPY docker/entrypoint.sh ./docker/entrypoint.sh

RUN mkdir -p /app/uploads /tmp/multiscope /var/lib/multiscope/xai-artifacts \
    && chown -R multiscope:multiscope /app /tmp/multiscope /var/lib/multiscope \
    && chmod +x /app/docker/entrypoint.sh

USER multiscope

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=development \
    DEBUG=true \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates build-essential pkg-config libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock pyproject.toml ./
# The CPU-only torch index avoids pulling the CUDA runtime, which would add
# several gigabytes to the image for no benefit in a CPU container.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" \
    && python -m pip install --no-cache-dir -c requirements.lock ".[dev,models,xai,glottal,mcp]" \
    && python -m pip check

RUN groupadd --system multiscope \
    && useradd --system --gid multiscope --home-dir /app --shell /usr/sbin/nologin multiscope

COPY app ./app
COPY mcp_server ./mcp_server
COPY scripts ./scripts
COPY tests ./tests
COPY run.py runtime_guard.py ./
COPY docker/entrypoint.sh ./docker/entrypoint.sh

# Check runtime imports during the build without local secrets or model loading.
RUN MULTISCOPE_DISABLE_DOTENV=1 python -c "from runtime_guard import ensure_supported_python; ensure_supported_python(); import torch, transformers, shap, xgboost; assert torch.version.cuda is None; from mcp.server.fastmcp import FastMCP; from app.models.real.glottal_compat import apply_disvoice_compatibility_patches; apply_disvoice_compatibility_patches(); from disvoice.glottal.glottal import Glottal; import app.main"

RUN mkdir -p /app/uploads /tmp/multiscope /var/lib/multiscope/xai-artifacts /var/cache/huggingface \
    && chown -R multiscope:multiscope /app /tmp/multiscope /var/lib/multiscope /var/cache/huggingface \
    && chmod +x /app/docker/entrypoint.sh

USER multiscope

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=10).read()"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "run.py"]

# MULTI-SCOPE Backend

The FastAPI backend requires Python 3.12 (`>=3.12,<3.13`) for development,
tests, Docker, and real-model/XAI inference. Python 3.11 and 3.13 are not
supported. The pinned SHAP 0.52.0 XAI dependency requires Python 3.12 or newer.

## Windows / VS Code Setup

Open a PowerShell terminal in VS Code and run from the repository root:

```powershell
cd backend/audio-service
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -c requirements.lock ".[dev]"
python run.py
```

For real audio models and XAI, install the optional dependencies before starting:

```powershell
python -m pip install -c requirements.lock ".[dev,models,xai,glottal]"
```

Select `.venv/Scripts/python.exe` with **Python: Select Interpreter** in VS Code.
Existing Python 3.11 environments must be recreated with Python 3.12; installing
Python 3.12 does not upgrade an existing virtual environment. Configure `.env`
from `.env.example` and provide the required database, model, and XAI artifacts
for the enabled features. Installing packages does not download trained artifacts.

## macOS Setup

Install Python 3.12 first if it is not already available:

```bash
brew install python@3.12
brew install ffmpeg
```

Both `ffmpeg` and `ffprobe` must be available on `PATH`. The application
readiness endpoint reports their availability.

Create a clean backend environment:

```bash
cd backend/audio-service

deactivate 2>/dev/null || true

rm -rf .venv

python3.12 -m venv .venv

source .venv/bin/activate

python --version

python -m pip install --upgrade pip

python -m pip install -c requirements.lock ".[dev]"

python run.py
```

Before installing dependencies, `python --version` should report Python 3.12.x.
Do not reuse a virtual environment created with another Python version.

Run the test suite from the activated environment:

```bash
pytest -v
```

## Docker Development

The development Docker image uses Python 3.12 and CPU-only PyTorch. It installs
all model, Glottal, XAI, development-test, and MCP dependencies, plus FFmpeg and
native build/runtime libraries. It runs as a non-root `multiscope` user. The
health check uses `/ready` to check prediction readiness, not just API liveness.

Run these commands from `backend/audio-service`, with Docker Desktop's Linux
engine running. Create `.env` from `.env.example` only if it does not already
exist; keep your working model/XAI paths and credentials. Compose loads `.env`
through `env_file` and overrides selected values for the
container network. Its resolved configuration can contain secrets; do not share
unredacted `docker compose config` output. Do not commit `.env`; it is ignored by
the Docker build context.

```bash
docker compose config --quiet
docker compose build backend
docker compose up --build
```

The Compose stack starts MongoDB and the backend, with all four real branches
required and real XAI enabled by default. Source directories are mounted read-only
and `python run.py` reloads application changes when `DEBUG=true`. Model artifacts
are mounted read-only from `audio-service/model_artifacts`; the directory must
exist. Real semantic artifacts must be present under `app/voice_xai/semantic/artifacts`
and match the configured manifest. Package installation does not create them.

The XLS-R backbone downloads on first use into a persistent `huggingface-cache`
volume. Internet access and sufficient memory/disk space are required for this
first run. Set `COMPOSE_SSL_XLSR_LOCAL_FILES_ONLY=true` only after the container's
cache has been populated; the Windows host cache is not used automatically.

Cloudinary enablement and storage policy inherit your `.env` settings unless
overridden with `COMPOSE_CLOUDINARY_STORAGE_ENABLED` or `COMPOSE_STORAGE_POLICY`.
Playback and stored-audio rerun require valid Cloudinary credentials and retained
audio. Clerk and optional AI narrative settings also come from `.env`. Keep
`DEV_AUTH_BYPASS` restricted to local development. Real model loading alone does
not establish research eligibility.

Wait for `Application startup complete.`, then check `http://localhost:8000/ready`
and `http://localhost:8000/docs`. The health check allows five minutes for initial
startup; a slow first download can take longer, so inspect the logs:

```bash
docker compose logs -f backend
docker compose exec backend python -m pytest -q
```

To run the optional MCP stdio process against the same database and mounted
artifacts, after building the image:

```bash
docker compose --profile mcp run --rm -T mcp
```

MCP clients must provide valid API-key credentials. This process is separate
from the HTTP server and does not expose an additional HTTP port.

Stop the development stack with `docker compose down`. Named volumes preserve
MongoDB records, XAI evidence, and the downloaded backbone across ordinary stops.
If you need to point Compose at an external MongoDB instance, use
`COMPOSE_MONGODB_URI` and `COMPOSE_MONGODB_DATABASE` rather than the production
application variable names. Compose-specific overrides use the `COMPOSE_`
prefix, for example `COMPOSE_STORAGE_POLICY` and
`COMPOSE_CLOUDINARY_STORAGE_ENABLED`, to avoid accidentally reusing production
application secrets.

Production and research deployments must set:

- `APP_ENV=production` or `APP_ENV=research`
- `DEBUG=false`
- non-wildcard `ALLOWED_ORIGINS`
- `MONGODB_URI` and `MONGODB_DATABASE`
- Clerk issuer, JWKS URL, audience, and authorized parties
- `STORAGE_POLICY=required`
- Cloudinary cloud name, API key, API secret, and audio folder

The settings validator fails closed for production/research when these values
are missing or unsafe.

## Dependency Reproducibility

`pyproject.toml` is the canonical dependency declaration. Runtime
dependencies live in `[project].dependencies`; development, integration-test,
and MCP adapter dependencies live in `[project.optional-dependencies]`.

`requirements.lock` pins the packages listed in it for Python 3.12 environments;
it is a constraints file, not a complete lock of every optional dependency.
Do not edit it opportunistically during feature work.
Refresh it only in a dedicated dependency update task.

Common installs:

```bash
python -m pip install -c requirements.lock .
python -m pip install -c requirements.lock ".[dev]"
python -m pip install -c requirements.lock ".[integration-test]"
python -m pip install -c requirements.lock ".[mcp]"
```

Use `pyproject.toml` with `requirements.lock` for local and Docker installs.
The Docker image installs CPU-only PyTorch from its dedicated index first, then
the `dev`, `models`, `xai`, `glottal`, and `mcp` groups with these constraints.
The build checks dependency consistency and runtime imports.

System dependencies:

- FFmpeg and ffprobe are required for audio inspection, validation, decoding,
  and reproducible preprocessing metadata.
- No declared Python package currently requires libsndfile. If future work adds
  `soundfile`, deployment images must include libsndfile or use a wheel that
  bundles the required native library.

## Prediction Job Execution

The current MVP creates a queued prediction record, then executes it inline
through `InlinePredictionJobRunner`. This keeps POST `/api/v1/predictions`
synchronous for now while separating prediction creation, execution, and status
retrieval. Blocking media inspection, FFmpeg decoding, NumPy preprocessing,
model branch inference, and fusion run in a bounded process-local executor
controlled by `MAX_CONCURRENT_PREDICTIONS=2`; MongoDB, Clerk, Cloudinary, and
HTTP orchestration remain asynchronous. Do not use FastAPI background tasks for
durable prediction execution: those jobs can be lost during server restarts.

Thread cancellation cannot forcibly terminate arbitrary Python, NumPy, FFmpeg,
or future PyTorch work that is already running. A timeout marks the prediction
failed and releases the request path, but the underlying worker thread may
finish later. Real model deployment should replace the inline runner boundary
with a durable worker process and explicit model-runtime cancellation strategy.

The deprecated compatibility route `POST /api/v1/voice/predict` is disabled by
default with `ENABLE_LEGACY_ANONYMOUS_PREDICTION=false`. It is never available
in production. If explicitly enabled for local development, it still requires a
verified Clerk user and delegates to the same prediction submission service and
job runner used by `POST /api/v1/predictions`; it must not be used as an
anonymous prediction path.

Audio upload endpoints use an ASGI request-body limit before route processing
and retain the streamed per-file ingestion limit. The default request-body cap
is `MAX_REQUEST_BODY_MB=27`, allowing multipart overhead above
`MAX_UPLOAD_SIZE_MB=25` while rejecting substantially oversized bodies.

The shared audio preprocessing contract is `audio-preprocessing-v2`: FFmpeg
decodes validated audio to mono float32 PCM at 16 kHz, decoded audio must be at
least `MIN_AUDIO_DURATION_SECONDS=1.0`, and model-agnostic windows are generated
once for all branches with `MODEL_WINDOW_DURATION_SECONDS=6.0`,
`MODEL_WINDOW_OVERLAP_SECONDS=1.0`, no trimming, and deterministic right-zero
padding. Branch-specific LFCC/SSL/glottal feature extraction remains inside
future branch adapters.

The in-process runner now has both global and per-principal backpressure:
`MAX_CONCURRENT_PREDICTIONS`, `MAX_QUEUED_PREDICTIONS`, and
`MAX_ACTIVE_PREDICTIONS_PER_PRINCIPAL`. HTTP rate limiting is also enforced per
process with `RATE_LIMIT_*`, `PREDICTION_RATE_LIMIT_PER_WINDOW`, and
`API_KEY_CREATION_RATE_LIMIT_PER_WINDOW`. In production, these backend limits
should complement edge/provider rate limits rather than replace them.

When deploying behind Nginx, configure the proxy with the same outer body cap:

```nginx
client_max_body_size 27m;
```

Future Redis plus worker migration point: replace the dependency-provided
`PredictionJobRunner` with a durable queue producer, then run a separate worker
that consumes queued prediction IDs and calls the existing execution service.
The stored prediction statuses already support the worker flow:
`queued`, `validating`, `storing`, `processing`, `completed`, and `failed`.

## Model Integration Runtime

The runtime supports four canonical branches:

| Internal branch | Public model name | Implementation |
| --- | --- | --- |
| `lfcc_cnn_tcn` | `cnn_acoustic` | LFCC CNN/TCN |
| `aasist` | `aasist` | AASIST-Light V2 |
| `ssl_sequence` | `ssl_wavlm_xlsr` | XLS-R with a Mamba head |
| `glottal` | `glottal_features` | Selected acoustic/glottal features with a trained classifier |

Each branch supports `real`, `dummy`, or `disabled` mode. `Settings` feeds
`app.models.factory.ModelFactory`, `app.models.registry.ModelRegistry`, and
`app.services.voice_service.VoiceService`. Enabled modes and required branches
come from configuration; do not infer them from the presence of source files.
Dummy scores are placeholders and cannot support research claims.

Real inference requires the `models` extra; real Glottal inference also requires
`glottal`. Install `xai` for real semantic explanations. Trained checkpoints and
supporting artifacts must be supplied separately from the Python packages.

Checkpoint paths resolve relative to `MODEL_ROOT_DIR`. The settings default is
`../model_artifacts`, relative to `audio-service`, which resolves to
`backend/model_artifacts`. If artifacts are inside `audio-service/model_artifacts`,
set `MODEL_ROOT_DIR=model_artifacts`. `app/models` contains Python implementations,
not trained checkpoints. Use the checkpoint filenames configured in `.env.example`
or explicitly configure the matching artifacts for your deployment.

The SSL loader uses a pretrained XLS-R backbone through `transformers` plus the
trained Mamba head. It supports a cached backbone with
`SSL_XLSR_LOCAL_FILES_ONLY=true`. Its pure-PyTorch selective scan avoids requiring
the CUDA-only `mamba-ssm` package. The current waveform frontend uses a six-second,
16 kHz center-crop/pad input with zero-mean/peak normalization.

Model loading can be configured for startup or lazy execution. Invalid required
checkpoints can fail startup; do not treat a Uvicorn reloader message as proof
that the application is ready. Wait for `Application startup complete.` and
inspect `/ready` and `/api/v1/voice/models/health`.

Fusion supports the frozen detector contract, the constrained four-branch V3
artifacts, and configured legacy fallback. The selected path depends on supplied
artifacts and settings. Contributing branches, weights, threshold, fallback, and
research blockers are exposed in the fusion result. Supply the matching fusion
artifacts rather than assuming a fixed branch count or equal weights.

Preprocessing and class-mapping verification flags are separate from runtime
availability. A loaded real model may still report `research_ready: false` or
an unverified-result warning. Enable verification flags only after validating
the corresponding trained artifact and preprocessing pipeline. Production and
research settings enforce additional verification requirements.

Device selection supports CPU, CUDA, MPS, and automatic selection, with explicit
CPU fallback settings. Inference timeouts and bounded thread execution do not
forcibly terminate an underlying model operation.

## Voice XAI

Voice XAI runs downstream of a persisted classifier result. An explanation failure
does not change a completed prediction into a failed prediction.

For real explanations, enable `XAI_ENABLED=true`, set `XAI_MODE=real`, and set
`XAI_SEMANTIC_BUNDLE_DIR` to the imported model bundle directory. Relative
paths resolve from `audio-service`. The semantic service uses a 165-feature
teacher surrogate with raw-score Tree SHAP.

All semantic features use one full cleaned waveform with no six-second crop.
This requires a retrained `shared-clean-fullclip-audio-v2` model; the previous
model bundle is rejected. Follow [the retraining steps](docs/semantic-retraining.md)
to rebuild inputs with the portable backend extractor.

To use a retrained model, export its model folder and run from `audio-service`:

```powershell
.\.venv\Scripts\python.exe -m app.voice_xai.semantic.import_bundle --source "D:\Models\new-surrogate"
```

Restart the backend afterwards. The command validates and atomically activates
an immutable bundle; no code edits or manual hash updates are required. Current
model version, decision threshold and optional fidelity R² follow the imported
metadata. R² targets never gate runtime. See [the model update guide](docs/semantic-pipeline.md).

`XAI_MODE` supports only `disabled` and `real`. There is no mock semantic
pipeline or generated SHAP fallback. A semantic service that has not been
initialized is reported as unavailable; a run records `semantic_service_unavailable`
without replacing real evidence with synthetic values.

Temporal explanations consume reduced evidence from the real XLS-R classifier
pass. Available attention windows are combined using cosine weighting. Regions
use only the per-clip 90th percentile of the resulting float32 density (linear
quantile, float32 cutoff, `>=` comparison), with no density floor or threshold
artifact. The explanation and visualization use the same threshold and regions.
Window bounds and displayed regions use source-audio timestamps, including the
classifier's center-crop offset. P90 uses only bins covered by original-pass
evidence. Uncovered time-series and spectrogram frames are `null`; the frontend
leaves the normal spectrogram untouched there and does not bridge curve gaps.
Ties and region postprocessing can make coverage differ from 10%; flat maps can
select all covered bins. Attention regions are model-focus evidence, not ground-truth
manipulated-audio labels. Optional AI narrative settings require their own
provider configuration.

New private temporal artifacts use `xlsr-temporal-evidence-v3-source-time`.
Earlier crop-relative artifacts cannot be reused safely because they lack the
source offset; rerun prediction to obtain new original-pass evidence.

Owner-scoped endpoints live under
`/api/v1/me/predictions/{prediction_id}/explanation`. Private artifact downloads
verify ownership and integrity. See the [Voice XAI module guide](app/voice_xai/README.md)
and [classifier-to-XAI contract](app/voice_xai/CLASSIFIER_XAI_CONTRACT.md) for the
current handoff, lifecycle, and evidence boundaries.

## Cloudinary Storage Consistency

Audio storage uses deterministic authenticated Cloudinary public IDs in the
form `multiscope/audio/{owner_user_id}/{audio_id}`. Configure both
`CLOUDINARY_SDK_TIMEOUT_SECONDS` for the SDK call and
`CLOUDINARY_UPLOAD_TIMEOUT_SECONDS` for the outer application wait. If the outer
timeout fires, the result is ambiguous because cancelling the asyncio waiter
does not forcibly stop the underlying SDK worker thread. The storage adapter
therefore reconciles the deterministic public ID and deletes orphaned assets
when the prediction has not successfully claimed them. The reconciliation
function is safe to rerun from a later maintenance job.

MongoDB and Cloudinary do not share an atomic transaction. The backend records
sanitised failure states and performs compensating cleanup where practical, but
operators should treat cross-system reconciliation as an explicit operational
responsibility.

## MCP Adapter

The `mcp_server` package is a separate deployable adapter for controlled MCP
integrations. Its tools call the same service layer used by FastAPI instead of
duplicating prediction, history, validation, storage, model, or fusion logic.

Initial MCP authentication uses existing MULTI-SCOPE API keys and maps MCP tool
access to the existing scopes:

```text
multiscope_create_prediction  -> prediction:create
multiscope_get_prediction     -> prediction:read
multiscope_list_predictions   -> prediction:list
multiscope_get_model_status   -> prediction:read
multiscope_delete_prediction  -> prediction:delete
```

For the first controlled implementation, prediction creation supports only
bounded `small_audio_payload_base64` submissions for short clips and tests. This
is not intended for normal 3-minute audio. Remote public MCP authorization and
large-audio upload handoff require a later OAuth-compatible implementation with
durable signed upload references.

Run the MCP adapter in a separate process after installing the optional MCP
dependencies in that environment:

```bash
python -m pip install -c requirements.lock ".[mcp]"
python -m mcp_server.server
```

The MCP process has its own startup and shutdown lifecycle. Startup requires
`MONGODB_URI`, connects to MongoDB, applies the same collection/index policy,
and creates one shared tool context containing reusable repositories, storage,
model registry, voice service, job runner, and concurrency semaphore. Shutdown
closes MongoDB and reusable prediction-runner resources.

## Documentation

See [Swagger testing guide](docs/setup/swagger-real-world-testing.md) for manual
API checks. The module guide and contract above describe the current XAI runtime.

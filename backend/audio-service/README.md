# MULTI-SCOPE Backend

The FastAPI backend requires Python 3.11. Python 3.12 is also supported by the
project metadata, but Python 3.11 is the documented development environment.

## macOS Setup

Install Python 3.11 first if it is not already available:

```bash
brew install python@3.11
brew install ffmpeg
```

Both `ffmpeg` and `ffprobe` must be available on `PATH`. The application
readiness endpoint reports their availability.

Create a clean backend environment:

```bash
cd backend

deactivate 2>/dev/null || true

rm -rf .venv

python3.11 -m venv .venv

source .venv/bin/activate

python --version

python -m pip install --upgrade pip

pip install -c requirements.lock ".[dev]"

python run.py
```

Before installing dependencies, `python --version` should report Python 3.11.x.
Do not reuse a virtual environment created with Python 3.9.

Run the test suite from the activated environment:

```bash
pytest -v
```

## Docker Development

The backend includes a Docker image for local deployment rehearsal. The image
uses Python 3.11, installs FFmpeg/ffprobe, runs as a non-root `multiscope`
user, and exposes `/health` as the container health check.

Copy `.env.example` to `.env` if you want local interpolation values before
starting Compose. The compose file does not pass `.env` wholesale into the
container, which avoids accidentally echoing Cloudinary, Clerk, or MongoDB
secrets during `docker-compose config`. Do not commit `.env`; it is ignored by
the Docker build context.

```bash
cp .env.example .env
docker compose up --build
```

The Compose stack starts MongoDB and the backend. For local Compose, the
default `STORAGE_POLICY=optional` plus `CLOUDINARY_STORAGE_ENABLED=false` allows
predictions to continue without Cloudinary audio storage. Playback URLs and
rerun-from-stored-audio are unavailable when audio is not successfully stored.
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

`requirements.lock` is an exact constraints file for reproducible Python 3.11
and 3.12 environments. Do not edit it opportunistically during feature work.
Refresh it only in a dedicated dependency update task.

Common installs:

```bash
pip install -c requirements.lock .
pip install -c requirements.lock ".[dev]"
pip install -c requirements.lock ".[integration-test]"
pip install -c requirements.lock ".[mcp]"
```

`requirements.txt` is kept only as a compatibility wrapper for tools that still
expect that filename. New documentation and CI should use `pyproject.toml` plus
`requirements.lock`.

System dependencies:

- FFmpeg and ffprobe are required for audio inspection, validation, decoding,
  and reproducible preprocessing metadata.
- No declared Python package currently requires libsndfile. If future work adds
  `soundfile`, deployment images must include libsndfile or use a wheel that
  bundles the required native library.

## Repository Layout

The authoritative repository should be the parent repository:

```text
deepfake-voice-classification-system/.git
```

The nested backend repository:

```text
deepfake-voice-classification-system/backend/.git
```

is a migration artifact. Keeping both active makes status, ignores, CI paths,
and commits ambiguous. Do not delete Git metadata until the backend history has
been backed up and reviewed.

Safe migration plan:

```bash
cd ..
git -C backend status --short
git -C backend branch --show-current
git -C backend log --oneline --decorate -n 20
tar -czf backend-git-backup-$(date +%Y%m%d-%H%M%S).tgz backend/.git
git status --short
```

After confirming the backup and reviewing any backend-only commits that must be
preserved, remove only the nested metadata directory:

```bash
rm -rf backend/.git
git status --short
```

Then stage the backend files from the parent repository and make the parent
repository the only commit source. The virtual environment remains ignored by
both repository configurations and must not be committed.

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

Each branch runs in one of three modes — `real`, `dummy`, or `disabled`. The
factory path is:

```text
Settings
  -> app.models.factory.ModelFactory
  -> app.models.registry.ModelRegistry
  -> app.services.voice_service.VoiceService
```

Canonical internal branch names are `lfcc_cnn_tcn`, `aasist`, `ssl_sequence`,
and `glottal`. Existing public model names such as `cnn_acoustic`,
`ssl_wavlm_xlsr`, and `glottal_features` are preserved in API responses for
backward compatibility.

### Current stage: three real branches

The CNN acoustic, AASIST-Light, and SSL (XLS-R + Mamba) branches load trained
checkpoints and run real inference. Glottal has no trained artifact and is
`disabled`: it executes nothing, returns `BranchStatus.skipped`, and is
excluded from fusion. `disabled` is distinct from `failed` — the former is a
configured absence, the latter a branch that should have worked.

Real mode requires the `models` extra (`pip install ".[models]"`, which brings
PyTorch and, for SSL, `transformers`) plus a valid checkpoint under
`MODEL_ROOT_DIR` with a supported extension (`.pt`, `.pth`, `.ckpt`, `.onnx`,
`.safetensors`). CNN/AASIST checkpoints load once at startup, are moved to the
resolved device, and run under `torch.inference_mode()` with `strict=True` +
`weights_only=True` loading; an architecture mismatch fails loudly rather than
being masked by `strict=False`, which would leave layers randomly initialised.

`MODEL_ROOT_DIR` defaults to `../model_artifacts`, the repository-root
model-artifact directory (for example,
`../model_artifacts/xlsr_mamba_asvspoof2019_best.pt` when
running from `backend/`). `app/models` contains Python model implementations,
not checkpoint files.

**SSL is not listed in `REQUIRED_MODEL_BRANCHES`.** It contributes to fusion
like any other branch when it succeeds, but a missing checkpoint, a missing
`transformers` install, a Hugging Face Hub outage on first startup, or an
inference error only excludes that one branch from that one prediction — it
can never fail the whole request. This is deliberate: SSL's extra runtime
dependency (network access on first load, ~1.2 GB download) makes it more
likely to be unavailable in some deployments than CNN/AASIST, and the
non-negotiable architectural rule is that a single branch's absence must
never take down prediction.

#### SSL (XLS-R + Mamba) branch specifics

The deployment artifact
(`model_artifacts/xlsr_mamba_asvspoof2019_best.pt`, ~4.5 MB) is
**not** a self-contained checkpoint. It is a metadata package:

```text
architecture              "XLSRMambaClassifier"
xlsr_model_name           "facebook/wav2vec2-large-xlsr-53"  (fetched via Hugging Face Hub, cached under ~/.cache/huggingface)
xlsr_frozen                True
xlsr_hidden_size           1024
mamba_dim                  256
num_classes                 2
label_mapping               {"bonafide": 0, "spoof": 1}
trainable_model_state       state dict for projection + 2x Mamba blocks + norm + classifier ONLY
best_epoch / best_dev_eer / final_eval_metrics   research metadata, never used as runtime thresholds
```

`app/models/real/ssl_sequence_inference.py` loads the pretrained backbone by
name via `transformers.AutoModel.from_pretrained(...)`, freezes it
(`requires_grad_(False)`, `.eval()`), reconstructs the trainable head via
`app/models/architectures/xlsr_mamba.py`, and restores
`trainable_model_state` with an explicit strict check (every artifact key
must exist in the reconstructed module with a matching shape; every key that
`load_state_dict` leaves "missing" must be a frozen `xlsr.*` backbone
parameter, never part of the trainable head) — the same "loud failure over
silent partial load" policy CNN/AASIST use, adapted for a partial state dict.

**Why the Mamba blocks are a pure-PyTorch reimplementation, not the
`mamba-ssm` package:** `mamba-ssm` and its `causal-conv1d` dependency ship
CUDA-only build backends. On a non-CUDA target (confirmed on Apple Silicon
macOS: `pip install --no-build-isolation --no-deps mamba-ssm` crashes at the
metadata-generation step because its `setup.py` unconditionally probes
`torch.version.cuda`, which is `None`) they cannot install at all. The
checkpoint's `trainable_model_state` key names (`in_proj`, `conv1d`,
`x_proj`, `dt_proj`, `out_proj`, `A_log`, `D`) are exactly `mamba_ssm`'s own
parameter names, confirming the *architecture* — `xlsr_mamba.py` reimplements
the same published selective-scan recurrence in plain PyTorch (the same
reference/"slow path" formula `mamba_ssm` itself ships to validate its fused
CUDA kernel against) rather than the fused kernel. Verified against the real
artifact: every one of its 28 `trainable_model_state` tensors loads onto the
reconstructed module with `strict=True`-equivalent shape/key checking and
zero unexpected/dropped keys (see
`tests/test_ssl_sequence_integration.py::test_reconstructed_architecture_state_dict_matches_real_artifact_exactly`).
On an Apple Silicon CPU, one forward pass (XLS-R backbone + both Mamba
blocks) measured ~6–8 s.

Preprocessing (`app/models/preprocessing/ssl_waveform.py`) is a dedicated
front end — a peer of the CNN/AASIST front ends, not a fork of either — for
the training-documented contract: 16 kHz mono (shared with CNN/AASIST, no
resampling divergence), crop/pad to exactly 96,000 samples (6 s), a same-length
attention mask so padding never dilutes the pooled representation, and
zero-mean/unit-variance normalisation over the real (non-padded) samples only,
matching `Wav2Vec2FeatureExtractor(do_normalize=True)`'s convention.

### Verification gate

The CNN/AASIST checkpoints shipped without a training notebook, feature
configuration, or recorded label map, and both networks accept any input
shape — so a wrong feature pipeline produces confident, wrong answers instead
of an error. SSL is different in kind: its artifact documents its own
preprocessing and label map, so there is nothing to reverse-engineer — but its
Mamba blocks run through a from-scratch pure-PyTorch reimplementation (see
above) that has not yet been cross-checked on this runtime against the
documented ASVspoof2019 LA evaluation (EER 0.0568). Both situations reduce to
the same policy: explicit settings, gated by explicit attestations, default
false.

```env
CNN_PREPROCESSING_VERIFIED=false
CNN_CLASS_MAPPING_VERIFIED=false
AASIST_PREPROCESSING_VERIFIED=false
AASIST_CLASS_MAPPING_VERIFIED=false
SSL_PREPROCESSING_VERIFIED=false
SSL_CLASS_MAPPING_VERIFIED=false
```

Until **both** flags are true for a branch, it still runs and returns real
model output, but `research_result` is false, fusion reports
`unverified_branch_contributed`, and the application refuses to start in
`production`/`research`. For SSL specifically, flip these only after
re-running the documented ASVspoof2019 LA evaluation against this pure-PyTorch
scan and confirming the reported EER. See
`MULTI_SCOPE_CNN_AASIST_Real_Model_Integration_Report.md` for how the CNN/AASIST
defaults were derived.

Fusion additionally reports `system_stage` and `research_blockers`. A result is
research eligible only when no placeholder contributed, every contributing
branch is attested, and the full designed branch set participated — so a
two-branch result can never be presented as four-branch performance.

Device policy: `MODEL_DEVICE_POLICY` supports `cpu`, `cuda`, `mps`, and `auto`
(CUDA, then MPS, then CPU). PyTorch is imported for capability detection only
when a real branch is enabled. **CPU is the default deliberately** — both models
are small enough that MPS kernel-launch overhead dominates; AASIST-Light
measured ~30 ms on CPU versus ~192 ms on MPS. CUDA fallback is explicit via
`MODEL_ALLOW_CPU_FALLBACK`.

Model lifecycle supports load-once semantics, `startup` or `lazy` loading,
safe unload on FastAPI/MCP shutdown, branch-level health, and bounded
per-branch inference timeout. The current isolation level is thread/task-level:
a timed-out blocking adapter may continue internally until its worker returns.
True crash/OOM isolation still requires process-level workers later.

Fusion is runtime-configured with `FUSION_METHOD`,
`FUSION_DECISION_THRESHOLD`, `FUSION_MIN_SUCCESSFUL_BRANCHES`,
`FUSION_WEIGHT_*`, `FUSION_REQUIRED_BRANCHES`, and `FUSION_CONFIG_VERSION`.
These defaults are development-safe infrastructure defaults; they are not
research-calibrated weights.

### Running an SSL smoke test

Hermetic tests (architecture reconstruction against the real artifact,
preprocessing, loader failure handling, fusion contribution/isolation) run in
the default `pytest` invocation whenever
`model_artifacts/xlsr_mamba_asvspoof2019_best.pt`
is present:

```bash
pytest tests/test_ssl_sequence_integration.py -q
```

The one true end-to-end test (real Hugging Face backbone download + real
forward pass) is marked `integration` + `network` like the Mongo/Cloudinary
tests, and needs the `models` extra installed:

```bash
pip install ".[models]"
pytest tests/test_ssl_sequence_integration.py -m "integration and network" -q
```

For a full-stack check, set `SSL_MODEL_MODE=real` and `SSL_MODEL_PATH=xlsr_mamba_asvspoof2019_best.pt`
(already the defaults in `.env.example`), start the app, and confirm
`GET /ready` reports `components.models.branches[].branch_name == "ssl_sequence"`
with `mode: "real"`, `is_loaded: true`, `ready: true`.

## Voice XAI attention localisation

The temporal XAI component uses the real XLS-R + Mamba branch. It requests
eager XLS-R attention from the trained
classifier, computes residual attention-rollout density for each six-second
window, combines overlapping windows with cosine edge weighting, and applies
the fixed PartialSpoof v1.2 DEV threshold in
`app/voice_xai/temporal/calibrations/partialspoof_v1_2_dev_attention_threshold.json`.
The resulting regions are candidate evidence, never a ground-truth fake-audio
percentage.

For the real temporal path, configure:

```env
XAI_ENABLED=true
XAI_MODE=real
MODEL_ROOT_DIR=../model_artifacts
SSL_MODEL_MODE=real
SSL_MODEL_PATH=xlsr_mamba_asvspoof2019_best.pt
```

`XAI_MODE=real` loads the checked-in v4 semantic XGBoost model together with
its hash-verified column-order, threshold, and training-metadata artifacts.
The semantic extraction reproduces the notebook's 148-feature recipe. A
completed run remains research-ineligible until the classifier verification
attestations and the required model evaluation policy are available. Submit a new prediction
after enabling XAI; the service retains that request's decoded audio only long
enough for its post-persistence XAI worker. A later manual retry cannot recover
the raw attention input from the persisted prediction alone.

Swagger exposes the owner-scoped lifecycle under
`/api/v1/me/predictions/{prediction_id}/explanation`, including the temporal
result and private visualization artifact endpoint.

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
pip install ".[mcp]"
python -m mcp_server.server
```

The MCP process has its own startup and shutdown lifecycle. Startup requires
`MONGODB_URI`, connects to MongoDB, applies the same collection/index policy,
and creates one shared tool context containing reusable repositories, storage,
model registry, voice service, job runner, and concurrency semaphore. Shutdown
closes MongoDB and reusable prediction-runner resources.

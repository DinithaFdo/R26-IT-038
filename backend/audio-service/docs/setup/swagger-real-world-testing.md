# Swagger Real-World Testing Guide

This guide explains how to test the backend manually through Swagger UI without
changing authentication, model, storage, or persistence behavior.

## 1. Activate the backend environment

From the repository root:

```bash
cd backend
source .venv/bin/activate
python --version
```

Use Python 3.11 or 3.12. The backend runtime guard rejects Python 3.13.

## 2. Start the backend

```bash
python run.py
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

OpenAPI JSON:

```text
http://127.0.0.1:8000/openapi.json
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## 3. Authenticate in Swagger

For local manual development only, you may avoid pasting a real Clerk token by
using the controlled development bypass:

```text
APP_ENV=development
DEV_AUTH_BYPASS=true
```

Both values are required. `DEV_AUTH_BYPASS=true` by itself must not disable
authentication in production, staging, test, or unknown environments. The safe
example default is `DEV_AUTH_BYPASS=false`.

For user-owned routes, click **Authorize** in Swagger and paste a Clerk session
JWT value. Swagger sends it as:

```text
Authorization: Bearer <token>
```

For third-party API-key routes, use the separate API-key bearer scheme with the
existing key format:

```text
Authorization: Bearer msk_live_...
```

Do not paste real tokens into tickets, reports, screenshots, or source files.

## 4. Test service liveness

Run `GET /health`.

Expected category of behavior: HTTP 200 when the API process is alive. This
does not prove MongoDB, storage, model readiness, or research readiness.

## 5. Test dependency readiness

Run `GET /ready`.

Inspect:

- `prediction_ready`
- `research_ready`
- `components.audio_tools`
- `components.mongodb`
- `components.storage`
- `components.models`
- `components.voice_xai`

Loaded/ready means the backend can run as configured. It is not the same as
research-validated or production-ready.

## 6. Test model health

Run `GET /api/v1/voice/models/health`.

Inspect each branch:

- `branch_name`
- `model_name`
- `mode`
- `is_loaded`
- `ready`
- `research_ready`
- `checkpoint_valid`
- `model_version`

For AASIST-Light V2, `research_ready: false` is not automatically a runtime
failure. It means the current verification policy has not marked the branch as a
research result.

## 7. Submit a prediction

Run `POST /api/v1/predictions`.

Use multipart form-data:

- `file`: choose a local audio file.
- `source_type`: `dashboard_upload` for a normal file upload.
- `client_filename`: optional display filename.
- `idempotency_key`: optional owner-scoped retry key. Leave it empty for
  normal manual testing -- every request without a key creates a new
  prediction.

Idempotency rules when you do send a key:

- Use a new unique value for every new audio submission, for example the output
  of `uuidgen`.
- Reuse the same value only to retry the exact same logical request. That
  replays the stored prediction instead of running inference twice.
- The logical request is `source_type` + `client_filename` + the uploaded
  filename, scoped to the authenticated owner. Reusing one key with any of
  those changed returns `409 idempotency_conflict`.
- Swagger UI keeps whatever you typed into the form between `Execute` presses,
  so clear the `idempotency_key` box (or paste a fresh UUID) before uploading a
  different file. With `DEV_AUTH_BYPASS=true` every local request shares the
  single `dev-local-user` owner, so keys stay claimed in MongoDB across
  restarts.
- If you hit `409 idempotency_conflict` locally, the `details` object names the
  fields that changed and the prediction the key is already bound to.

The backend flow is:

```text
upload -> validation -> audio preprocessing -> model branches -> fusion -> persistence -> optional XAI handoff
```

Supported categories include WAV, FLAC, MP3, M4A, AAC, Opus, OGG, and
audio-only WebM when ffprobe confirms a supported audio stream.

## 8. Inspect the prediction response

Look for:

- `prediction_id`
- `request_id`
- `status`
- `audio`
- `branches`
- `fusion.prediction`
- `fusion.probabilities`
- `research_eligible`

Do not assume exact scores from ad hoc real-world files. Scores depend on the
model checkpoints, preprocessing, and input audio.

## 9. Optionally test XAI

After a completed prediction, use its `prediction_id` with:

- `POST /api/v1/me/predictions/{prediction_id}/explanation`
- `GET /api/v1/me/predictions/{prediction_id}/explanation`
- `GET /api/v1/me/predictions/{prediction_id}/explanation/temporal`
- `GET /api/v1/me/predictions/{prediction_id}/explanation/semantic`
- `GET /api/v1/me/predictions/{prediction_id}/explanation/report`

Explanation work is independent of the classifier result. XAI unavailable,
disabled, failed, or partial states should not change a completed prediction
into a failed prediction.

## 10. Real-World Test Data Checklist

Test with:

- Known bonafide/human recording: should return a valid completed prediction or
  a controlled model/storage readiness error.
- Known spoof/synthetic recording: should return a valid completed prediction or
  a controlled model/storage readiness error.
- Short valid audio: should either complete or return the existing controlled
  minimum-duration validation behavior.
- MP3/FLAC compressed audio: should be accepted when ffprobe confirms supported
  audio-only content within configured limits.
- Near-silent audio: should return the existing controlled silent/unusable audio
  error category.
- Corrupted or invalid audio: should return the existing controlled corrupted or
  unsupported audio error category.

Do not treat one manual upload as an accuracy benchmark. Use labelled evaluation
sets for research metrics.

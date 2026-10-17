# Voice XAI through MCP

Voice classification and Voice XAI use the same `mcp_server.server` stdio server.
No extra HTTP port or second MCP connection is needed. XAI tools call the existing
`VoiceXaiOrchestrator`; the Clerk-authenticated HTTP routes remain available.

## Runtime prerequisites

Use Python 3.12 and install the model/XAI/MCP extras, or build the existing Docker
image, which already includes them. Keep the working `.env` and trained artifacts.

```powershell
python -m pip install -c requirements.lock ".[models,xai,glottal,mcp]"
```

The MCP process needs reachable MongoDB, the same `API_KEY_HASH_SECRET` and database
used when issuing its API key, the classifier artifacts, and these XAI settings:

```dotenv
XAI_ENABLED=true
XAI_MODE=real
XAI_SEMANTIC_MODEL_DIR=app/voice_xai/semantic/artifacts
```

The model directory must contain the matching semantic exports and manifest.
Real semantic initialization rejects missing or incompatible artifacts at startup.
Manual explanation creation/retry needs retained source audio and working storage
credentials. Temporal evidence comes from the original real XLS-R inference pass;
manual retries reuse its private saved evidence and do not rerun the classifier.
An older prediction without that evidence can have unavailable temporal analysis.
Optional narratives also need the existing `XAI_NARRATIVE_*` settings.

## Create an API key

As the prediction's owner, call `POST /api/v1/me/api-keys` using a Clerk bearer token
(for example, through Swagger). A key for classification and XAI can use:

```json
{
  "name": "Voice classification and XAI MCP",
  "scopes": [
    "prediction:create",
    "prediction:read",
    "prediction:list",
    "xai:create",
    "xai:read",
    "xai:retry"
  ]
}
```

Save the returned `api_key` securely: it is displayed once. Existing keys with
only `prediction:*` scopes do not authorize XAI tools; issue a new key with the
required XAI scopes. A read-only explanation client only needs `xai:read`.
Do not commit keys or insert them into shared prompts or client configuration.

Every tool verifies the token and scope before accessing XAI. Ownership is taken
from the verified API key. There is no `owner_user_id` tool argument, and tools
cannot read or retry another owner's explanations or artifacts. Trigger/retry
rate limits apply per owner within each MCP process.

## Connect a local stdio client

For a client that lets you set the working directory, use:

| Setting | Value |
| --- | --- |
| Transport | stdio |
| Command | Absolute path to `audio-service/.venv/Scripts/python.exe` on Windows |
| Arguments | `-m mcp_server.server` |
| Working directory | Absolute path to `backend/audio-service` |

For Docker, first start the local database and build the image:

```powershell
cd "D:\Research Project\R26-IT-038\backend\audio-service"
docker compose build backend
docker compose up -d mongodb
```

Then configure the client to launch Docker with the following command/arguments.
This example uses the `mcpServers` format supported by many desktop clients;
other clients use their own settings format. Adjust the absolute project path
if the checkout moves.

```json
{
  "mcpServers": {
    "multiscope-voice": {
      "command": "docker",
      "args": [
        "compose",
        "--project-directory",
        "D:/Research Project/R26-IT-038/backend/audio-service",
        "--profile", "mcp",
        "run", "--rm", "--no-deps", "-T", "mcp"
      ]
    }
  }
}
```

`--no-deps` assumes MongoDB is already running. Keep `-T`: MCP uses stdin/stdout
without a terminal. The client launches and retains this process; typing the
launch command in an unrelated terminal does not connect that terminal to the
client. The existing Compose volumes share model artifacts, XAI artifacts, and
the Hugging Face cache between the API and MCP services.

## Tool workflow

The server exposes five classification tools and nine XAI tools:

| Tool | Scope | Purpose |
| --- | --- | --- |
| `multiscope_create_explanation` | `xai:create` | Queue or reuse an explanation for an owned completed prediction |
| `multiscope_get_explanation` | `xai:read` | Poll the latest run and read its available evidence |
| `multiscope_get_temporal_explanation` | `xai:read` | Read temporal evidence |
| `multiscope_get_semantic_explanation` | `xai:read` | Read semantic/SHAP evidence |
| `multiscope_get_explanation_report` | `xai:read` | Read the deterministic combined report |
| `multiscope_get_explanation_narrative` | `xai:read` | Read the optional AI narrative |
| `multiscope_get_explanation_artifact` | `xai:read` | Read a public artifact reference's verified bytes as base64, up to 1 MiB |
| `multiscope_retry_explanation` | `xai:retry` | Queue a new explanation run while preserving previous evidence |
| `multiscope_retry_explanation_narrative` | `xai:retry` | Retry only an eligible narrative, using persisted evidence |

For each XAI tool, supply the key as `api_token` and the existing `prediction_id`:

```json
{
  "api_token": "<your-api-key>",
  "prediction_id": "<completed-prediction-id>"
}
```

The artifact tool additionally takes `artifact_id`, selected from the explanation's
public `artifacts` list. It never accepts filesystem paths or serves the internal
temporal retry artifact.

1. Create an owned prediction through MCP or the existing application.
2. Wait until classification is completed. With XAI enabled, MCP classification
   automatically enqueues explanation work using its original inference evidence.
3. Call `multiscope_get_explanation`. For a completed prediction without a run,
   call `multiscope_create_explanation`, then poll the get tool while queued/running.
4. Inspect component statuses/errors and read the available evidence/report.
   A run can be partial or blocked; do not assume all components succeeded.
5. Use the explicit retry tools when needed. Repeated create calls reuse the latest
   run unless it failed; a failed run requires the retry tool.

Tool results retain the adapter's `{ "ok", "request_id", "data", "error" }`
envelope. Check `ok` even when the MCP transport call itself succeeds. Controlled
errors include `permission_denied`, `not_found`, `conflict` (including pending
evidence or unavailable replay), `rate_limited`, `service_unavailable`, and
`payload_too_large`. Component failure never changes a completed classifier result
or grants research eligibility.

## Verification

Run the MCP adapter and XAI integration tests from `audio-service`:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_mcp_adapter.py tests/test_mcp_xai.py -q
```

These tests use in-memory repositories and the real XAI orchestrator plus the
pinned MCP SDK, with controlled analysis services. They cover owner/scope checks,
tool discovery/calls, automatic original-evidence handoff, polling, retries,
artifact bounds, and startup/shutdown. They do not validate real model accuracy or
replace a deployment test against MongoDB, storage, and the trained artifacts.

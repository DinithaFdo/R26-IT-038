export type EndpointDoc = {
  method: "GET" | "POST" | "DELETE";
  path: string;
  purpose: string;
  auth: string;
  scopes?: string;
  params?: string[];
  query?: string[];
  body?: string[];
  response: string;
  errors: string[];
  curl: string;
};

export const endpointDocs: EndpointDoc[] = [
  {
    method: "GET",
    path: "/health",
    purpose: "Backend liveness check.",
    auth: "Public",
    response: "{ status: string }",
    errors: ["500 internal server error"],
    curl: "curl http://127.0.0.1:8000/health",
  },
  {
    method: "GET",
    path: "/ready",
    purpose: "Backend readiness including prediction, research, audio tools, MongoDB, storage, runner, and models.",
    auth: "Public",
    response: "ReadinessResponse",
    errors: ["503 one or more dependencies unavailable"],
    curl: "curl http://127.0.0.1:8000/ready",
  },
  {
    method: "POST",
    path: "/api/v1/predictions",
    purpose: "Create an authenticated dashboard upload or browser-recording prediction.",
    auth: "Clerk session bearer token",
    body: ["multipart file", "source_type=dashboard_upload|live_recording", "client_filename?", "idempotency_key?"],
    response: "PredictionSubmissionResponse",
    errors: ["400 invalid audio", "401 auth failed", "409 idempotency conflict", "413 too large", "415 unsupported format", "422 validation", "503 unavailable"],
    curl: "curl -X POST \"$API_BASE_URL/api/v1/predictions\" -H \"Authorization: Bearer YOUR_SESSION_TOKEN\" -F \"file=@sample.wav\" -F \"source_type=dashboard_upload\" -F \"idempotency_key=demo-001\"",
  },
  {
    method: "GET",
    path: "/api/v1/predictions/{prediction_id}/status",
    purpose: "Fetch owner-scoped prediction job status.",
    auth: "Clerk session bearer token",
    params: ["prediction_id"],
    response: "PredictionJobStatusResponse",
    errors: ["401 auth failed", "404 not found", "500 internal server error"],
    curl: "curl \"$API_BASE_URL/api/v1/predictions/PREDICTION_ID/status\" -H \"Authorization: Bearer YOUR_SESSION_TOKEN\"",
  },
  {
    method: "GET",
    path: "/api/v1/me/predictions",
    purpose: "List authenticated user's prediction history.",
    auth: "Clerk session bearer token",
    query: ["page", "limit", "status", "source_type", "prediction_label", "created_from", "created_to"],
    response: "PredictionHistoryListResponse",
    errors: ["401 auth failed"],
    curl: "curl \"$API_BASE_URL/api/v1/me/predictions?page=1&limit=20\" -H \"Authorization: Bearer YOUR_SESSION_TOKEN\"",
  },
  {
    method: "GET",
    path: "/api/v1/me/predictions/{prediction_id}",
    purpose: "Fetch full owner-scoped prediction detail.",
    auth: "Clerk session bearer token",
    params: ["prediction_id"],
    response: "PredictionDetailResponse",
    errors: ["401 auth failed", "404 not found"],
    curl: "curl \"$API_BASE_URL/api/v1/me/predictions/PREDICTION_ID\" -H \"Authorization: Bearer YOUR_SESSION_TOKEN\"",
  },
  {
    method: "GET",
    path: "/api/v1/me/predictions/{prediction_id}/audio",
    purpose: "Create a short-lived signed playback URL for retained audio.",
    auth: "Clerk session bearer token",
    params: ["prediction_id"],
    response: "PredictionAudioPlaybackResponse",
    errors: ["401 auth failed", "404 not found or unavailable"],
    curl: "curl \"$API_BASE_URL/api/v1/me/predictions/PREDICTION_ID/audio\" -H \"Authorization: Bearer YOUR_SESSION_TOKEN\"",
  },
  {
    method: "POST",
    path: "/api/v1/me/predictions/{prediction_id}/rerun",
    purpose: "Create a new prediction from retained owner-scoped source audio.",
    auth: "Clerk session bearer token",
    params: ["prediction_id"],
    body: ["rerun_reason?"],
    response: "PredictionSubmissionResponse",
    errors: ["401 auth failed", "404 not found", "503 source audio unavailable"],
    curl: "curl -X POST \"$API_BASE_URL/api/v1/me/predictions/PREDICTION_ID/rerun\" -H \"Authorization: Bearer YOUR_SESSION_TOKEN\"",
  },
  {
    method: "DELETE",
    path: "/api/v1/me/predictions/{prediction_id}",
    purpose: "Soft-delete an owner-scoped prediction and retained audio when present.",
    auth: "Clerk session bearer token",
    params: ["prediction_id"],
    response: "PredictionDeleteResponse",
    errors: ["401 auth failed", "404 not found"],
    curl: "curl -X DELETE \"$API_BASE_URL/api/v1/me/predictions/PREDICTION_ID\" -H \"Authorization: Bearer YOUR_SESSION_TOKEN\"",
  },
  {
    method: "GET",
    path: "/api/v1/voice/models/health",
    purpose: "Report safe branch model health and dummy/real mode.",
    auth: "Public when model health is enabled",
    response: "ModelHealthResponse[]",
    errors: ["404 disabled"],
    curl: "curl \"$API_BASE_URL/api/v1/voice/models/health\"",
  },
  {
    method: "POST",
    path: "/api/v1/external/predictions",
    purpose: "Create a prediction using an API key.",
    auth: "API key bearer token",
    scopes: "prediction:create",
    body: ["multipart file", "client_filename?", "idempotency_key?"],
    response: "PredictionSubmissionResponse",
    errors: ["401 invalid API key", "403 missing scope", "413 too large", "415 unsupported format", "422 validation"],
    curl: "curl -X POST \"$API_BASE_URL/api/v1/external/predictions\" -H \"Authorization: Bearer YOUR_API_KEY\" -F \"file=@sample.wav\" -F \"idempotency_key=external-001\"",
  },
];

export const mcpTools = [
  {
    name: "multiscope_create_prediction",
    scope: "prediction:create",
    purpose: "Submit a bounded small audio payload for prediction through the backend service layer.",
    input: "api_token, filename?, content_type?, client_filename?, idempotency_key?, small_audio_payload_base64",
    output: "MCPToolResponse with compact PredictionSubmissionResponse data",
  },
  {
    name: "multiscope_get_prediction",
    scope: "prediction:read",
    purpose: "Retrieve one owner-scoped prediction detail.",
    input: "api_token, prediction_id",
    output: "MCPToolResponse with compact prediction detail",
  },
  {
    name: "multiscope_list_predictions",
    scope: "prediction:list",
    purpose: "List owner-scoped predictions with backend-supported pagination and filters.",
    input: "api_token, page?, limit?, status?, source_type?, prediction_label?, created_from?, created_to?",
    output: "MCPToolResponse with items, page, limit, has_next",
  },
  {
    name: "multiscope_get_model_status",
    scope: "prediction:read",
    purpose: "Fetch compact model branch mode and loaded status.",
    input: "api_token",
    output: "MCPToolResponse with models[]",
  },
  {
    name: "multiscope_delete_prediction",
    scope: "prediction:delete",
    purpose: "Delete one owner-scoped prediction.",
    input: "api_token, prediction_id",
    output: "MCPToolResponse with PredictionDeleteResponse data",
  },
];

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ApiKeyScope(str, Enum):
    prediction_create = "prediction:create"
    prediction_read = "prediction:read"
    prediction_list = "prediction:list"
    prediction_delete = "prediction:delete"
    audio_read = "audio:read"


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[ApiKeyScope] = Field(min_length=1)
    expires_at: datetime | None = None


class ApiKeyMetadata(BaseModel):
    api_key_id: str
    key_prefix: str
    name: str
    scopes: list[ApiKeyScope]
    created_at: datetime
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyCreateResponse(ApiKeyMetadata):
    api_key: str


class ApiKeyListResponse(BaseModel):
    items: list[ApiKeyMetadata]


class ApiKeyDeleteResponse(BaseModel):
    api_key_id: str
    revoked: bool

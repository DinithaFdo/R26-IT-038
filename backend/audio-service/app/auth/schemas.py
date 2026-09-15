from pydantic import BaseModel, Field


class AuthPrincipal(BaseModel):
    """Future authenticated caller context for Clerk users and API keys."""

    subject: str
    principal_type: str
    user_id: str | None = None
    organisation_id: str | None = None
    api_key_id: str | None = None
    scopes: list[str] = Field(default_factory=list)

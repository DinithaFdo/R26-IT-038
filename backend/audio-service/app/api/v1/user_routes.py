from fastapi import APIRouter, Depends

from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    summary="Get the authenticated application principal",
    description=(
        "Returns a sanitized authenticated principal derived from a verified "
        "Clerk session token. Raw JWT claims are never returned."
    ),
)
async def get_current_user(
    principal: AuthPrincipal = Depends(require_clerk_user),
) -> AuthPrincipal:
    return principal

from app.auth.schemas import AuthPrincipal


def principal_has_scope(principal: AuthPrincipal, required_scope: str) -> bool:
    return required_scope in principal.scopes


def require_scope(principal: AuthPrincipal, required_scope: str) -> None:
    if not principal_has_scope(principal, required_scope):
        raise PermissionError("Authenticated principal lacks the required scope.")

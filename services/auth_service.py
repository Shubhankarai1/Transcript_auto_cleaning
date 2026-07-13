from __future__ import annotations

from typing import Any

from supabase_client import get_supabase_admin_client, is_supabase_configured


class AuthError(Exception):
    pass


def verify_token(token: str) -> dict[str, Any]:
    if not is_supabase_configured():
        raise AuthError('Authentication is not configured')

    client = get_supabase_admin_client()
    try:
        response = client.auth.get_user(token)
        user_data = response.user
        if user_data is None:
            raise AuthError('Invalid token')
        return dict(user_data)
    except Exception as exc:
        raise AuthError(str(exc))

from __future__ import annotations

from supabase import Client, create_client

from config import get_env, require_env


SUPABASE_URL = get_env('SUPABASE_URL')
SUPABASE_ANON_KEY = get_env('SUPABASE_ANON_KEY')
SUPABASE_SERVICE_ROLE_KEY = get_env('SUPABASE_SERVICE_ROLE_KEY')


def is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY))


def create_supabase_client(use_service_role: bool = True) -> Client:
    if not SUPABASE_URL:
        require_env('SUPABASE_URL')

    key_name = 'SUPABASE_SERVICE_ROLE_KEY' if use_service_role else 'SUPABASE_ANON_KEY'
    api_key = SUPABASE_SERVICE_ROLE_KEY if use_service_role else SUPABASE_ANON_KEY

    if not api_key:
        require_env(key_name)

    return create_client(SUPABASE_URL, api_key)


def get_supabase_admin_client() -> Client:
    return create_supabase_client(use_service_role=True)

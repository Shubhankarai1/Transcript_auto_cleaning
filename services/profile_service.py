from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from supabase_client import get_supabase_admin_client, is_supabase_configured


UNAVAILABLE = object()


def is_available() -> bool:
    return is_supabase_configured()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_client():
    if not is_supabase_configured():
        return None
    try:
        return get_supabase_admin_client()
    except Exception:
        return None


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        'user_id': record.get('user_id'),
        'email': record.get('email'),
        'full_name': record.get('full_name'),
        'job_role': record.get('job_role'),
        'industry': record.get('industry'),
        'years_experience': record.get('years_experience'),
        'career_aspirations': record.get('career_aspirations'),
        'ai_learning_goals': record.get('ai_learning_goals'),
        'weekly_learning_availability': record.get('weekly_learning_availability'),
        'onboarding_completed': bool(record.get('onboarding_completed', False)),
        'created_at': record.get('created_at'),
        'updated_at': record.get('updated_at'),
    }


def _serialize_payload(user_id: str, email: str | None = None, full_name: str | None = None,
                       job_role: str | None = None, industry: str | None = None,
                       years_experience: int | None = None,
                       career_aspirations: str | None = None,
                       ai_learning_goals: str | None = None,
                       weekly_learning_availability: str | None = None,
                       onboarding_completed: bool = False) -> dict[str, Any]:
    return {
        'user_id': user_id,
        'email': email,
        'full_name': full_name,
        'job_role': job_role,
        'industry': industry,
        'years_experience': years_experience,
        'career_aspirations': career_aspirations,
        'ai_learning_goals': ai_learning_goals,
        'weekly_learning_availability': weekly_learning_availability,
        'onboarding_completed': onboarding_completed,
        'updated_at': _utc_now_iso(),
    }


def fetch_record(user_id: str) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return UNAVAILABLE
    try:
        response = client.table('profiles').select('*').eq('user_id', user_id).limit(1).execute()
    except Exception:
        return None

    data = response.data or []
    if not data:
        return None
    return dict(data[0])


def upsert_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return UNAVAILABLE
    try:
        response = client.table('profiles').upsert(payload).execute()
    except Exception:
        return None

    data = response.data or []
    if not data:
        return payload
    return dict(data[0])

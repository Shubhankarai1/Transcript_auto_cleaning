from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from routers.deps import get_current_user_optional
from services import profile_service


router = APIRouter(prefix='/v1/profile')


class ProfileBase(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    job_role: Optional[str] = None
    industry: Optional[str] = None
    years_experience: Optional[int] = None
    career_aspirations: Optional[str] = None
    ai_learning_goals: Optional[str] = None
    weekly_learning_availability: Optional[str] = None
    onboarding_completed: bool = False


class ProfileUpsertRequest(ProfileBase):
    user_id: Optional[str] = None


class ProfileResponse(ProfileUpsertRequest):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _authenticated_user_id(
    token_user: dict[str, Any] | None,
    explicit_user_id: str | None = None,
) -> str:
    """Resolve user identity: prefer the JWT-authenticated user, fall back to an explicit user_id."""
    if token_user:
        return token_user['id']
    if explicit_user_id:
        return explicit_user_id
    raise HTTPException(
        status_code=401,
        detail='Provide an Authorization header (Bearer token) or pass user_id explicitly.',
    )


@router.get('', response_model=ProfileResponse)
def get_profile(
    token_user: dict[str, Any] | None = Depends(get_current_user_optional),
    user_id: str = Query(default=None),
):
    resolved_id = _authenticated_user_id(token_user, user_id)
    record = profile_service.fetch_record(resolved_id)
    if record is profile_service.UNAVAILABLE:
        raise HTTPException(status_code=503, detail='Supabase is not configured')
    if record is None:
        raise HTTPException(status_code=404, detail='Profile not found')
    return profile_service.normalize_record(record)


@router.post('', response_model=ProfileResponse)
def create_profile(
    req: ProfileUpsertRequest,
    token_user: dict[str, Any] | None = Depends(get_current_user_optional),
):
    resolved_id = _authenticated_user_id(token_user, req.user_id)
    existing = profile_service.fetch_record(resolved_id)
    if existing is profile_service.UNAVAILABLE:
        raise HTTPException(status_code=503, detail='Supabase is not configured')
    if existing is not None:
        raise HTTPException(status_code=409, detail='Profile already exists')

    payload = profile_service._serialize_payload(
        user_id=resolved_id, email=req.email, full_name=req.full_name,
        job_role=req.job_role, industry=req.industry,
        years_experience=req.years_experience,
        career_aspirations=req.career_aspirations,
        ai_learning_goals=req.ai_learning_goals,
        weekly_learning_availability=req.weekly_learning_availability,
        onboarding_completed=req.onboarding_completed,
    )
    record = profile_service.upsert_record(payload)
    if record is profile_service.UNAVAILABLE:
        raise HTTPException(status_code=503, detail='Supabase is not configured')
    if record is None:
        raise HTTPException(status_code=500, detail='Failed to save profile')
    return profile_service.normalize_record(record)


@router.put('', response_model=ProfileResponse)
def update_profile(
    req: ProfileUpsertRequest,
    token_user: dict[str, Any] | None = Depends(get_current_user_optional),
):
    resolved_id = _authenticated_user_id(token_user, req.user_id)
    payload = profile_service._serialize_payload(
        user_id=resolved_id, email=req.email, full_name=req.full_name,
        job_role=req.job_role, industry=req.industry,
        years_experience=req.years_experience,
        career_aspirations=req.career_aspirations,
        ai_learning_goals=req.ai_learning_goals,
        weekly_learning_availability=req.weekly_learning_availability,
        onboarding_completed=req.onboarding_completed,
    )
    record = profile_service.upsert_record(payload)
    if record is profile_service.UNAVAILABLE:
        raise HTTPException(status_code=503, detail='Supabase is not configured')
    if record is None:
        raise HTTPException(status_code=500, detail='Failed to save profile')
    return profile_service.normalize_record(record)

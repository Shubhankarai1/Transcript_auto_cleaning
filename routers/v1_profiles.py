from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

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
    user_id: str


class ProfileResponse(ProfileUpsertRequest):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.get('', response_model=ProfileResponse)
def get_profile(user_id: str = Query(...)):
    record = profile_service.fetch_record(user_id)
    if record is profile_service.UNAVAILABLE:
        raise HTTPException(status_code=503, detail='Supabase is not configured')
    if record is None:
        raise HTTPException(status_code=404, detail='Profile not found')
    return profile_service.normalize_record(record)


@router.post('', response_model=ProfileResponse)
def create_profile(req: ProfileUpsertRequest):
    existing = profile_service.fetch_record(req.user_id)
    if existing is profile_service.UNAVAILABLE:
        raise HTTPException(status_code=503, detail='Supabase is not configured')
    if existing is not None:
        raise HTTPException(status_code=409, detail='Profile already exists')

    payload = profile_service._serialize_payload(
        user_id=req.user_id, email=req.email, full_name=req.full_name,
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
def update_profile(req: ProfileUpsertRequest):
    payload = profile_service._serialize_payload(
        user_id=req.user_id, email=req.email, full_name=req.full_name,
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

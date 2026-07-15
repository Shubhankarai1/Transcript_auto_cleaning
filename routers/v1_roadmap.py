from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from routers.deps import get_current_user
from services import roadmap_service


router = APIRouter(prefix='/v1/roadmap')


@router.post('/generate')
def generate_roadmap(
    body: dict[str, Any],
    user: dict[str, Any] = Depends(get_current_user),
):
    track_id = body.get('track_id')
    if not track_id:
        raise HTTPException(status_code=400, detail='track_id is required')

    industry = body.get('industry')
    job_role = body.get('job_role')

    roadmap = roadmap_service.generate_roadmap(track_id, industry=industry, job_role=job_role)
    record = roadmap_service.save_roadmap(user['id'], track_id, roadmap)

    if record is None:
        raise HTTPException(status_code=503, detail='Failed to save roadmap')

    return {
        'plan_id': record.get('id'),
        'roadmap': roadmap,
    }


@router.get('/active')
def get_active_roadmap(
    user: dict[str, Any] = Depends(get_current_user),
):
    record = roadmap_service.get_active_roadmap(user['id'])
    if record is None:
        raise HTTPException(status_code=404, detail='No active roadmap found')

    return {
        'plan_id': record.get('id'),
        'track': record.get('track'),
        'roadmap': record.get('roadmap_json'),
        'version': record.get('version'),
        'created_at': record.get('created_at'),
    }


@router.get('/{plan_id}')
def get_roadmap_by_id(
    plan_id: str,
    user: dict[str, Any] = Depends(get_current_user),
):
    record = roadmap_service.get_roadmap_by_id(plan_id, user['id'])
    if record is None:
        raise HTTPException(status_code=404, detail='Roadmap not found')

    return {
        'plan_id': record.get('id'),
        'track': record.get('track'),
        'roadmap': record.get('roadmap_json'),
        'version': record.get('version'),
        'created_at': record.get('created_at'),
    }

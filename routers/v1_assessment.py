from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from routers.deps import get_current_user
from services import assessment_service


router = APIRouter(prefix='/v1/assessment')


@router.get('/questions')
def get_questions():
    return {'questions': assessment_service.get_questions()}


@router.post('/submit')
def submit_assessment(
    body: dict[str, Any],
    user: dict[str, Any] = Depends(get_current_user),
):
    answers = body.get('answers', [])
    if not answers:
        raise HTTPException(status_code=400, detail='No answers provided')

    user_id = user['id']
    record = assessment_service.submit_assessment(user_id, answers)
    if record is None:
        raise HTTPException(status_code=503, detail='Assessment service unavailable')

    result = record.get('scored_result', {})
    return {
        'attempt_id': record.get('id'),
        'result': result,
    }


@router.get('/result')
def get_latest_result(
    user: dict[str, Any] = Depends(get_current_user),
):
    record = assessment_service.get_latest_assessment(user['id'])
    if record is None:
        raise HTTPException(status_code=404, detail='No assessment found')

    result = record.get('scored_result', {})
    return {
        'attempt_id': record.get('id'),
        'result': result,
        'created_at': record.get('created_at'),
    }


@router.get('/result/{attempt_id}')
def get_result_by_id(
    attempt_id: str,
    user: dict[str, Any] = Depends(get_current_user),
):
    record = assessment_service.get_assessment_by_id(attempt_id)
    if record is None:
        raise HTTPException(status_code=404, detail='Assessment not found')
    if record.get('user_id') != user['id']:
        raise HTTPException(status_code=403, detail='Not your assessment')

    result = record.get('scored_result', {})
    return {
        'attempt_id': record.get('id'),
        'result': result,
        'created_at': record.get('created_at'),
    }

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services import track_service


router = APIRouter(prefix='/v1/tracks')


@router.get('')
def list_tracks():
    return {'tracks': track_service.get_tracks()}


@router.get('/{track_id}')
def get_track(track_id: str):
    track = track_service.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f'Unknown track: {track_id}')
    return {'track': track}


@router.get('/{track_id}/modules')
def get_track_modules(
    track_id: str,
    role: str | None = Query(default=None),
):
    if track_service.get_track(track_id) is None:
        raise HTTPException(status_code=404, detail=f'Unknown track: {track_id}')

    modules = track_service.get_track_modules(track_id, role=role)
    if modules is None:
        raise HTTPException(status_code=404, detail=f'Unknown track: {track_id}')

    return {'track_id': track_id, 'modules': modules, 'total': len(modules)}


@router.get('/{track_id}/content')
def get_track_content(track_id: str):
    content = track_service.get_track_content(track_id)
    if content is None:
        raise HTTPException(status_code=404, detail=f'Unknown track: {track_id}')
    return content

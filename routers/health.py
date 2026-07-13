from fastapi import APIRouter

from supabase_client import is_supabase_configured


router = APIRouter()


@router.get('/')
def root():
    return {'status': 'API running'}


@router.get('/health')
def health():
    return {
        'status': 'healthy',
        'supabase_configured': is_supabase_configured(),
    }

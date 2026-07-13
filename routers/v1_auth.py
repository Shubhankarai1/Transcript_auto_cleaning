from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from routers.deps import get_current_user


router = APIRouter(prefix='/v1/auth')


@router.get('/me')
def get_me(user: dict[str, Any] = Depends(get_current_user)):
    return {
        'id': user.get('id'),
        'email': user.get('email'),
        'created_at': user.get('created_at'),
        'last_sign_in_at': user.get('last_sign_in_at'),
        'phone': user.get('phone'),
        'app_metadata': user.get('app_metadata', {}),
        'user_metadata': user.get('user_metadata', {}),
    }

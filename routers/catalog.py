from fastapi import APIRouter, HTTPException, Query

from services.catalog_service import LEVEL_ORDER, get_content_catalog


router = APIRouter()


@router.get('/levels')
def get_levels():
    catalog = get_content_catalog()
    return sorted(catalog.keys(), key=lambda level: LEVEL_ORDER.get(level, 99))


@router.get('/modules')
def get_modules(level: str | None = Query(default=None)):
    catalog = get_content_catalog()
    if level is None:
        all_modules = {module for modules in catalog.values() for module in modules}
        return sorted(all_modules)

    normalized_level = level.strip().lower()
    modules = catalog.get(normalized_level)
    if modules is None:
        raise HTTPException(status_code=404, detail=f'Unknown level: {level}')
    return sorted(modules.keys())


@router.get('/sessions')
def get_sessions(module: str = Query(...), level: str | None = Query(default=None)):
    catalog = get_content_catalog()
    normalized_module = module.strip().lower()

    if level is not None:
        normalized_level = level.strip().lower()
        modules = catalog.get(normalized_level)
        if modules is None:
            raise HTTPException(status_code=404, detail=f'Unknown level: {level}')
        sessions = modules.get(normalized_module)
        if sessions is None:
            raise HTTPException(status_code=404, detail=f"Unknown module '{module}' for level '{level}'")
        return sessions

    for modules in catalog.values():
        sessions = modules.get(normalized_module)
        if sessions is not None:
            return sessions

    raise HTTPException(status_code=404, detail=f'Unknown module: {module}')

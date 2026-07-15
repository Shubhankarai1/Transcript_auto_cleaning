from __future__ import annotations

from typing import Any

from services.catalog_service import get_content_catalog


TRACKS = {
    'foundations': {
        'id': 'foundations',
        'label': 'AI Foundations',
        'level': 'beginner',
        'description': 'Build foundational AI literacy — understand key concepts, prompt engineering, and responsible AI use.',
        'audience': 'Operations, HR, Finance Managers and professionals new to AI',
        'persona': 'The Explorer',
    },
    'practitioner': {
        'id': 'practitioner',
        'label': 'AI Practitioner',
        'level': 'intermediate',
        'description': 'Integrate AI into workflows — data analysis, RAG, and human-in-the-loop system design.',
        'audience': 'Customer-facing roles, Project leads, Non-tech Managers',
        'persona': 'The Workflow Architect',
    },
    'builder': {
        'id': 'builder',
        'label': 'AI Builder',
        'level': 'advanced',
        'description': 'Architect AI solutions — enterprise strategy, agentic frameworks, and model optimization.',
        'audience': 'Solutions Architects, Software Developers, Data Scientists',
        'persona': 'The Solutions Builder',
    },
}


ROLE_MODULE_MAP = {
    'foundations': {
        'common_modules': [
            'ai_foundations_curriculum',
            'prompt_engineering',
            'ai_ethics_safety_and_data_privacy',
        ],
        'role_modules': {
            'finance': 'finance_chatgpt_excel_skills',
            'hr': 'hr_ai_enhanced_jd_design_and_skills_gap_mapping',
            'operations': 'operations_process_mapping_and_automated_reporting',
        },
    },
    'practitioner': {
        'common_modules': [
            'ai_data_analysis_extracting_insights',
            'human_in_the_loop_designing_hybrid_systems',
        ],
        'role_modules': {
            'customer_facing': 'customer_facing_ai_sentiment_analysis_and_crm_integration',
            'project_management': 'project_management_predictive_resource_allocation_and_automated_risk_tracking',
        },
    },
    'builder': {
        'common_modules': [
            'cms',
            'map',
            'wdp',
        ],
        'role_modules': {},
    },
}


def get_tracks() -> list[dict[str, Any]]:
    return list(TRACKS.values())


def get_track(track_id: str) -> dict[str, Any] | None:
    return TRACKS.get(track_id)


def _get_module_sessions(catalog: dict, module_name: str) -> list[int]:
    for modules in catalog.values():
        sessions = modules.get(module_name)
        if sessions is not None:
            return sessions
    return []


def get_track_modules(track_id: str, role: str | None = None) -> list[dict[str, Any]] | None:
    track = TRACKS.get(track_id)
    if track is None:
        return None

    mapping = ROLE_MODULE_MAP.get(track_id)
    if mapping is None:
        return []

    catalog = get_content_catalog()
    result: list[dict[str, Any]] = []

    for module_name in mapping['common_modules']:
        sessions = _get_module_sessions(catalog, module_name)
        result.append({
            'module': module_name,
            'category': 'common',
            'sessions': sessions,
            'role_specific': False,
        })

    if role and role in mapping['role_modules']:
        role_module = mapping['role_modules'][role]
        sessions = _get_module_sessions(catalog, role_module)
        result.append({
            'module': role_module,
            'category': 'role_specific',
            'sessions': sessions,
            'role_specific': True,
            'role': role,
        })

    return result


def get_track_content(track_id: str) -> dict[str, Any] | None:
    track = TRACKS.get(track_id)
    if track is None:
        return None

    modules = get_track_modules(track_id)
    return {
        'track': track,
        'total_modules': len(modules) if modules else 0,
        'modules': modules or [],
    }

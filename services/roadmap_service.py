from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from services.track_service import TRACKS, ROLE_MODULE_MAP, _get_module_sessions
from services.catalog_service import get_content_catalog
from supabase_client import get_supabase_admin_client, is_supabase_configured

UNAVAILABLE = object()

WEEKLY_PLANS = {
    'foundations': [
        {
            'week': 1,
            'focus': 'AI Landscape & Terminology',
            'objectives': [
                'Understand the evolution of AI and key terminologies',
                'Distinguish between traditional AI, predictive AI, and GenAI',
                'Learn about LLMs, tokens, and context windows',
                'Identify which business problems are AI-solvable',
            ],
            'modules': ['ai_foundations_curriculum'],
            'sessions': [1],
            'estimated_hours': 4,
        },
        {
            'week': 2,
            'focus': 'Prompt Engineering',
            'objectives': [
                'Master prompt anatomy: roles, context, constraints',
                'Learn Few-Shot and Chain-of-Thought prompting',
                'Practice structuring complex tasks into prompt chains',
                'Develop skills for iterative prompt debugging',
            ],
            'modules': ['prompt_engineering'],
            'sessions': [1],
            'estimated_hours': 5,
        },
        {
            'week': 3,
            'focus': 'AI Ethics, Safety & Data Privacy',
            'objectives': [
                'Identify PII and confidential IP in AI workflows',
                'Understand Shadow AI risks and organizational policy',
                'Recognize prompt injection and data poisoning attacks',
                'Learn Human-in-the-Loop best practices',
            ],
            'modules': ['ai_ethics_safety_and_data_privacy'],
            'sessions': [1, 2],
            'estimated_hours': 5,
        },
        {
            'week': 4,
            'focus': 'Role-Specific Application',
            'objectives': [
                'Apply AI skills to your specific domain',
                'Complete a practical domain project',
                'Build a reusable workflow for your role',
            ],
            'modules': [],
            'sessions': [1],
            'estimated_hours': 6,
            'role_specific': True,
        },
    ],
    'practitioner': [
        {
            'week': 1,
            'focus': 'AI Data Analysis & Insights',
            'objectives': [
                'Learn data preparation and cleaning with AI',
                'Perform automated exploratory data analysis',
                'Identify trends and generate visualizations',
                'Synthesize data narratives for stakeholders',
            ],
            'modules': ['ai_data_analysis_extracting_insights'],
            'sessions': [1],
            'estimated_hours': 5,
        },
        {
            'week': 2,
            'focus': 'Human-in-the-Loop Systems',
            'objectives': [
                'Design decision checkpoints for AI workflows',
                'Build feedback loops for AI-human collaboration',
                'Create fallback protocols for low-confidence scenarios',
                'Implement accountability and auditing mechanisms',
            ],
            'modules': ['human_in_the_loop_designing_hybrid_systems'],
            'sessions': [1],
            'estimated_hours': 5,
        },
        {
            'week': 3,
            'focus': 'RAG & Internal Knowledge Integration',
            'objectives': [
                'Understand RAG architecture and lifecycle',
                'Learn knowledge base management best practices',
                'Implement source tracing and verification',
                'Design security and access control for RAG',
            ],
            'modules': [],
            'sessions': [],
            'estimated_hours': 5,
        },
        {
            'week': 4,
            'focus': 'Role-Specific Application',
            'objectives': [
                'Apply integration skills to your domain',
                'Complete a cross-functional AI project',
                'Design a hybrid workflow for your team',
            ],
            'modules': [],
            'sessions': [1],
            'estimated_hours': 6,
            'role_specific': True,
        },
    ],
    'builder': [
        {
            'week': 1,
            'focus': 'Contextual Reasoning & Multi-Agent Systems',
            'objectives': [
                'Understand agentic patterns and ReAct loops',
                'Learn multi-agent orchestration',
                'Design tool-calling and environment interaction',
                'Implement advanced reasoning paradigms',
            ],
            'modules': ['cms'],
            'sessions': [1, 2],
            'estimated_hours': 6,
        },
        {
            'week': 2,
            'focus': 'Advanced Agent Architecture',
            'objectives': [
                'Master planning strategies (CoT, Tree-of-Thoughts)',
                'Implement long-term memory for agents',
                'Design sandboxed execution environments',
                'Build fault-tolerant orchestration',
            ],
            'modules': ['cms'],
            'sessions': [3, 4],
            'estimated_hours': 6,
        },
        {
            'week': 3,
            'focus': 'Multi-Agent Planning & Workflows',
            'objectives': [
                'Design multi-step agent workflows',
                'Implement reflection and self-correction',
                'Handle complex state management',
                'Build production-grade agent chains',
            ],
            'modules': ['map'],
            'sessions': [2, 3],
            'estimated_hours': 6,
        },
        {
            'week': 4,
            'focus': 'Production Deployment & Workflow Optimization',
            'objectives': [
                'Optimize workflows for production',
                'Implement monitoring and evaluation',
                'Deploy and scale AI solutions',
                'Capstone project: end-to-end agent system',
            ],
            'modules': ['wdp'],
            'sessions': [1],
            'estimated_hours': 8,
        },
    ],
}


OBJECTIVE_LABELS = {
    'ai_foundations_curriculum': 'AI Foundations Curriculum',
    'prompt_engineering': 'Prompt Engineering',
    'ai_ethics_safety_and_data_privacy': 'AI Ethics, Safety & Data Privacy',
    'finance_chatgpt_excel_skills': 'Finance: ChatGPT & Excel Skills',
    'hr_ai_enhanced_jd_design_and_skills_gap_mapping': 'HR: JD Design & Skills-Gap Mapping',
    'operations_process_mapping_and_automated_reporting': 'Operations: Process Mapping & Reporting',
    'ai_data_analysis_extracting_insights': 'AI Data Analysis: Extracting Insights',
    'human_in_the_loop_designing_hybrid_systems': 'Human-in-the-Loop Design',
    'customer_facing_ai_sentiment_analysis_and_crm_integration': 'Customer Facing: Sentiment Analysis & CRM',
    'project_management_predictive_resource_allocation_and_automated_risk_tracking': 'Project Management: Resource Allocation & Risk',
    'cms': 'Contextual Reasoning for Multi-Agent Systems',
    'map': 'Multi-Agent Planning & Workflow Design',
    'wdp': 'Workflow Design & Optimization',
}


def _map_role_to_industry(industry: str | None) -> str | None:
    if not industry:
        return None
    industry_lower = industry.lower()
    if 'financ' in industry_lower:
        return 'finance'
    if 'hr' in industry_lower or 'human resource' in industry_lower:
        return 'hr'
    if 'operation' in industry_lower or 'manufactur' in industry_lower:
        return 'operations'
    if 'customer' in industry_lower or 'sales' in industry_lower or 'market' in industry_lower:
        return 'customer_facing'
    if 'project' in industry_lower or 'management' in industry_lower:
        return 'project_management'
    if 'tech' in industry_lower or 'it' in industry_lower or 'software' in industry_lower or 'developer' in industry_lower:
        return None
    return None


def _enrich_week(
    week: dict[str, Any],
    role_module_name: str | None,
    catalog: dict,
) -> dict[str, Any]:
    week = dict(week)
    enriched_sessions = []

    for mn in week.get('modules', []):
        sessions = _get_module_sessions(catalog, mn)
        label = OBJECTIVE_LABELS.get(mn, mn.replace('_', ' ').title())
        enriched_sessions.append({
            'module': mn,
            'label': label,
            'session_numbers': week.get('sessions', []),
            'available_sessions': sessions,
        })

    if week.get('role_specific') and role_module_name:
        sessions = _get_module_sessions(catalog, role_module_name)
        label = OBJECTIVE_LABELS.get(role_module_name, role_module_name.replace('_', ' ').title())
        enriched_sessions.append({
            'module': role_module_name,
            'label': label,
            'session_numbers': [1],
            'available_sessions': sessions,
            'role_specific': True,
        })

    week['sessions_detail'] = enriched_sessions
    return week


def generate_roadmap(
    track_id: str,
    industry: str | None = None,
    job_role: str | None = None,
) -> dict[str, Any]:
    track = TRACKS.get(track_id)
    track_info = {
        'id': track_id,
        'label': track['label'] if track else track_id,
        'level': track['level'] if track else 'beginner',
        'persona': track['persona'] if track else '',
    }

    role = _map_role_to_industry(industry or job_role)
    mapping = ROLE_MODULE_MAP.get(track_id, {})
    role_module_name = None
    if role and mapping:
        role_module_name = mapping.get('role_modules', {}).get(role)

    catalog = get_content_catalog()
    weekly_plan = WEEKLY_PLANS.get(track_id, [])
    weeks = [_enrich_week(w, role_module_name, catalog) for w in weekly_plan]

    total_hours = sum(w.get('estimated_hours', 0) for w in weeks)

    return {
        'track': track_info,
        'industry': industry,
        'role': role,
        'role_module': role_module_name,
        'estimated_duration': '4 weeks (30 days)',
        'estimated_total_hours': total_hours,
        'weeks': weeks,
    }


def _get_client():
    if not is_supabase_configured():
        return None
    try:
        return get_supabase_admin_client()
    except Exception:
        return None


def save_roadmap(user_id: str, track_id: str, roadmap: dict[str, Any]) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return None

    try:
        client.table('roadmap_plans').update({'is_active': False}).eq('user_id', user_id).execute()
    except Exception:
        pass

    record = {
        'user_id': user_id,
        'track': track_id,
        'roadmap_json': roadmap,
        'version': 1,
        'is_active': True,
    }
    try:
        response = client.table('roadmap_plans').insert(record).execute()
        data = response.data or []
        if data:
            return dict(data[0])
        return record
    except Exception:
        return None


def get_active_roadmap(user_id: str) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return None

    try:
        response = (
            client.table('roadmap_plans')
            .select('*')
            .eq('user_id', user_id)
            .eq('is_active', True)
            .order('created_at', desc=True)
            .limit(1)
            .execute()
        )
        data = response.data or []
        if not data:
            return None
        return dict(data[0])
    except Exception:
        return None


def get_roadmap_by_id(plan_id: str, user_id: str) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return None

    try:
        response = (
            client.table('roadmap_plans')
            .select('*')
            .eq('id', plan_id)
            .eq('user_id', user_id)
            .limit(1)
            .execute()
        )
        data = response.data or []
        if not data:
            return None
        return dict(data[0])
    except Exception:
        return None

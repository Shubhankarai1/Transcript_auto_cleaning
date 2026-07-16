from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from supabase import create_client

from config import get_env


SESSIONS_DIR = Path(__file__).resolve().parent / 'sessions'
SESSIONS_DIR.mkdir(exist_ok=True)


API_BASE_URL = get_env(
    'API_BASE_URL',
    'https://iitm-curriculem-intelligence-layer.onrender.com',
)
SUPABASE_URL = get_env('SUPABASE_URL')
SUPABASE_ANON_KEY = get_env('SUPABASE_ANON_KEY')
REQUEST_TIMEOUT = 60
MAX_HISTORY_MESSAGES = 10
LEVEL_LABELS = {
    'beginner': 'Foundations',
    'intermediate': 'Intermediate',
    'advanced': 'Advanced',
}
MODULE_LABELS = {
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

MODULE_DESCRIPTIONS = {
    'ai_foundations_curriculum': 'Core AI concepts, terminology, and real-world applications for beginners.',
    'prompt_engineering': 'Craft effective prompts and interact with LLMs for business tasks.',
    'ai_ethics_safety_and_data_privacy': 'Ethical AI use, bias mitigation, data privacy, and responsible deployment.',
    'finance_chatgpt_excel_skills': 'Apply AI to financial workflows — Excel automation, analysis, and reporting.',
    'hr_ai_enhanced_jd_design_and_skills_gap_mapping': 'Use AI for job descriptions, skill-gap mapping, and HR operations.',
    'operations_process_mapping_and_automated_reporting': 'Leverage AI for process mapping, workflow automation, and reporting.',
    'ai_data_analysis_extracting_insights': 'Extract actionable insights from data using AI-driven analysis.',
    'human_in_the_loop_designing_hybrid_systems': 'Design systems where humans and AI collaborate in decision-making.',
    'customer_facing_ai_sentiment_analysis_and_crm_integration': 'Integrate AI into customer workflows with sentiment analysis and CRM.',
    'project_management_predictive_resource_allocation_and_automated_risk_tracking': 'Use AI for resource allocation, risk tracking, and project oversight.',
    'cms': 'Build multi-agent systems with contextual reasoning, memory, and tool-use capabilities.',
    'map': 'Design multi-agent planning workflows with query expansion and reranking.',
    'wdp': 'Optimize LLM workflows with LangGraph, LangFlow, tool calling, and production deployment.',
}

MODULE_SESSION_COUNTS = {
    'ai_foundations_curriculum': 1,
    'prompt_engineering': 1,
    'ai_ethics_safety_and_data_privacy': 2,
    'finance_chatgpt_excel_skills': 1,
    'hr_ai_enhanced_jd_design_and_skills_gap_mapping': 1,
    'operations_process_mapping_and_automated_reporting': 1,
    'ai_data_analysis_extracting_insights': 1,
    'human_in_the_loop_designing_hybrid_systems': 1,
    'customer_facing_ai_sentiment_analysis_and_crm_integration': 1,
    'project_management_predictive_resource_allocation_and_automated_risk_tracking': 1,
    'cms': 4,
    'map': 2,
    'wdp': 5,
}

MODULE_TRACK_MAP = {
    'ai_foundations_curriculum': 'foundations',
    'prompt_engineering': 'foundations',
    'ai_ethics_safety_and_data_privacy': 'foundations',
    'finance_chatgpt_excel_skills': 'foundations',
    'hr_ai_enhanced_jd_design_and_skills_gap_mapping': 'foundations',
    'operations_process_mapping_and_automated_reporting': 'foundations',
    'ai_data_analysis_extracting_insights': 'practitioner',
    'human_in_the_loop_designing_hybrid_systems': 'practitioner',
    'customer_facing_ai_sentiment_analysis_and_crm_integration': 'practitioner',
    'project_management_predictive_resource_allocation_and_automated_risk_tracking': 'practitioner',
    'cms': 'builder',
    'map': 'builder',
    'wdp': 'builder',
}
INDUSTRIES = ['', 'Technology', 'Healthcare', 'Finance', 'Education',
              'Manufacturing', 'Retail', 'Media', 'Consulting', 'Other']
EXP_RANGES = ['', '0–1 years', '1–3 years', '3–5 years', '5–10 years', '10+ years']
HOURS_OPTIONS = ['', '1–2 hours', '3–5 hours', '5–10 hours', '10+ hours']


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _supabase_client():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error('Supabase is not configured.')
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _auth_headers() -> dict[str, str] | None:
    token = st.session_state.get('auth_token')
    if token:
        return {'Authorization': f'Bearer {token}'}
    return None


def do_signup(email: str, password: str) -> str | None:
    try:
        client = _supabase_client()
        resp = client.auth.sign_up({'email': email, 'password': password})
        return None if resp.user else 'Signup failed.'
    except Exception as exc:
        return str(exc)


def _handle_auth_error() -> None:
    do_logout()
    st.error('Your session has expired. Please sign in again.')
    st.rerun()


def do_login(email: str, password: str) -> str | None:
    try:
        client = _supabase_client()
        resp = client.auth.sign_in_with_password({'email': email, 'password': password})
        if resp.session:
            access_token = resp.session.access_token
            refresh_token = resp.session.refresh_token or ''
            sid = _save_session(access_token, refresh_token, email)
            _set_sid(sid)
            st.session_state.auth_token = access_token
            st.session_state.user_email = email
            st.session_state._sid = sid
            return None
        return 'Login failed.'
    except Exception as exc:
        return str(exc)


def do_logout() -> None:
    sid = st.session_state.pop('_sid', None) or _get_sid()
    if sid:
        _delete_session(sid)
    _clear_sid()
    for key in ('auth_token', 'user_email', 'profile',
                'chat_history', 'chat_scope_key', '_sid'):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return bool(st.session_state.get('auth_token'))


# ---------------------------------------------------------------------------
# Session persistence (survives browser refresh)
# ---------------------------------------------------------------------------

def _get_sid() -> str | None:
    return st.query_params.get('sid') or None


def _set_sid(sid: str) -> None:
    st.query_params['sid'] = sid


def _clear_sid() -> None:
    st.query_params.clear()


def _session_path(sid: str) -> Path:
    return SESSIONS_DIR / f'{sid}.json'


def _save_session(access_token: str, refresh_token: str, email: str) -> str:
    sid = uuid.uuid4().hex
    data = {'access_token': access_token, 'refresh_token': refresh_token, 'email': email}
    _session_path(sid).write_text(json.dumps(data, indent=2), encoding='utf-8')
    return sid


def _load_session(sid: str) -> dict | None:
    path = _session_path(sid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None


def _delete_session(sid: str) -> None:
    path = _session_path(sid)
    if path.exists():
        path.unlink()


def _recover_session() -> None:
    sid = _get_sid()
    if not sid:
        return
    data = _load_session(sid)
    if data is None:
        _clear_sid()
        return
    st.session_state.auth_token = data['access_token']
    st.session_state.user_email = data.get('email', '')
    st.session_state._sid = sid
    saved_page = st.query_params.get('p')
    if saved_page in ('dashboard', 'assessment', 'mentor', 'profile'):
        st.session_state.page = saved_page


# ---------------------------------------------------------------------------
# Profile / Onboarding helpers
# ---------------------------------------------------------------------------

def fetch_profile() -> dict | None:
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = requests.get(f'{API_BASE_URL}/v1/profile', headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (401, 403):
            _handle_auth_error()
            return None
    except requests.RequestException:
        pass
    return None


def save_profile(payload: dict) -> dict | None:
    headers = _auth_headers()
    if not headers:
        st.error('Not authenticated.')
        return None
    try:
        resp = requests.put(f'{API_BASE_URL}/v1/profile', json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (401, 403):
            _handle_auth_error()
            return None
        st.error(f'Error {resp.status_code}: {resp.text[:200]}')
    except requests.ConnectionError:
        st.error('Cannot reach the backend.')
    except requests.Timeout:
        st.error('Request timed out.')
    except requests.RequestException as exc:
        st.error(str(exc))
    return None


def onboarding_done() -> bool:
    p = st.session_state.get('profile')
    return bool(p and p.get('onboarding_completed'))


# ---------------------------------------------------------------------------
# Helpers for exp value mapping
# ---------------------------------------------------------------------------

def _exp_to_value(label: str) -> int:
    return {'0–1 years': 1, '1–3 years': 2, '3–5 years': 4, '5–10 years': 7, '10+ years': 12}.get(label, 0)


def _value_to_exp(value: int) -> str:
    if value <= 1:
        return '0–1 years'
    if value <= 3:
        return '1–3 years'
    if value <= 5:
        return '3–5 years'
    if value <= 10:
        return '5–10 years'
    return '10+ years'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown("""
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        .stApp { background: #fafafa; }
        .block-container { max-width: 1100px; padding-top: 1.5rem; }
        html { font-size: 125%; }

        .auth-card {
            max-width: 420px; margin: 0 auto; padding: 2.5rem 2rem;
            background: #fff; border-radius: 14px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
        }
        .auth-card h2 { font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 0.25rem; }
        .auth-card .subtitle { color: #6b7280; font-size: 0.92rem; margin-bottom: 1.5rem; }
        .auth-card .stButton > button { border-radius: 8px; font-weight: 600; }

        .card {
            background: #fff; border-radius: 12px; padding: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-1px); }
        .card-sm { padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
        .card-medium { max-width: 700px; margin: 1rem auto; }

        h1, h2, h3 { color: #111827; }
        .stButton > button { border-radius: 8px; font-weight: 600; transition: opacity 0.15s ease; }
        .stButton > button:hover { opacity: 0.9; }

        div[data-testid="stSidebar"] { background: #fff; }
        div[data-testid="stSidebar"] .stButton > button { width: 100%; }
        div[data-testid="stSidebar"] label { font-size: 1.5rem; padding: 0.4rem 0; }
        div[data-testid="stSidebar"] .stRadio > label { display: none; }
        header[data-testid="stHeader"] { display: none; }
        hr { margin: 2rem 0; }
        .main > div { animation: fadeIn 0.2s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        div[data-testid="stChatMessage"] { margin-bottom: 0.5rem; }
        div[data-testid="stChatMessageContent"] { line-height: 1.6; }

        .badge-foundations { background: rgba(16,185,129,0.12); color: #10b981; }
        .badge-practitioner { background: rgba(245,158,11,0.12); color: #f59e0b; }
        .badge-builder { background: rgba(239,68,68,0.12); color: #ef4444; }
        .badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
        .badge-pill { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 8px; font-size: 0.72rem; font-weight: 500; }

        .page-title { margin-bottom: 0.25rem; }
        .page-subtitle { color: #6b7280; margin-bottom: 1.5rem; }
        .text-muted { color: #6b7280; }
        .text-light { color: #9ca3af; }
        .stat-number { font-size: 2rem; font-weight: 800; }
        .stat-label { color: #9ca3af; }
        .score-number { font-size: 3rem; font-weight: 800; }
        .flex-center { display: flex; align-items: center; gap: 0.75rem; }
        .mt-1 { margin-top: 0.75rem; }
        .mb-1 { margin-bottom: 0.75rem; }

        @media (max-width: 768px) {
            .block-container { max-width: 100%; padding: 1rem; }
            .card { padding: 1.25rem; }
        }
        @media (max-width: 480px) {
            html { font-size: 100%; }
        }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def auth_page() -> None:
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<h2>AI Mentor</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Sign in to access your personalized AI learning assistant.</div>",
        unsafe_allow_html=True,
    )

    tab_login, tab_signup = st.tabs(['Sign In', 'Sign Up'])

    with tab_login:
        with st.form('login_form'):
            email = st.text_input('Email')
            password = st.text_input('Password', type='password')
            if st.form_submit_button('Sign In', use_container_width=True, type='primary'):
                if not email or not password:
                    st.error('Email and password are required.')
                else:
                    err = do_login(email, password)
                    if err:
                        st.error(err)
                    else:
                        st.rerun()

    with tab_signup:
        with st.form('signup_form'):
            email = st.text_input('Email')
            password = st.text_input('Password', type='password', help='At least 6 characters')
            if st.form_submit_button('Create Account', use_container_width=True, type='primary'):
                if not email or not password:
                    st.error('Email and password are required.')
                elif len(password) < 6:
                    st.error('Password must be at least 6 characters.')
                else:
                    err = do_signup(email, password)
                    if err:
                        st.error(err)
                    else:
                        st.success('Check your email for a confirmation link, then sign in.')

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def onboarding_page() -> None:
    profile = st.session_state.get('profile', {})
    step = st.session_state.get('onboarding_step', 1)

    st.markdown("<div class='card card-medium'>", unsafe_allow_html=True)
    st.markdown("<h3>Set up your profile</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280;'>Help us personalize your learning.</p>", unsafe_allow_html=True)

    st.progress(step / 3, text=f'Step {step} of 3')

    if step == 1:
        name = st.text_input('Full Name', value=profile.get('full_name', '') or '')
        industry = st.selectbox('Industry', options=INDUSTRIES, index=0)
        if st.button('Next', use_container_width=True, type='primary'):
            if not name.strip():
                st.error('Full name is required.')
            else:
                profile.update({'full_name': name.strip(), 'industry': industry})
                st.session_state.profile = profile
                st.session_state.onboarding_step = 2
                st.rerun()

    elif step == 2:
        exp_label = _value_to_exp(profile.get('years_experience') or 0)
        exp_idx = EXP_RANGES.index(exp_label) if exp_label in EXP_RANGES else 0
        exp = st.selectbox('Years of Experience', options=EXP_RANGES, index=exp_idx)
        goal = st.text_input('Career Goal', value=profile.get('career_aspirations', '') or '',
                             placeholder='e.g. Become an AI engineer')
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button('Back', use_container_width=True):
                st.session_state.onboarding_step = 1
                st.rerun()
        with c2:
            if st.button('Next', use_container_width=True, type='primary'):
                profile.update({'years_experience': _exp_to_value(exp), 'career_aspirations': goal.strip()})
                st.session_state.profile = profile
                st.session_state.onboarding_step = 3
                st.rerun()

    elif step == 3:
        ai_goals = st.text_input('AI Learning Goals', value=profile.get('ai_learning_goals', '') or '',
                                 placeholder='e.g. Learn RAG and LLM deployment')
        avail_idx = HOURS_OPTIONS.index(profile.get('weekly_learning_availability', '')) \
            if profile.get('weekly_learning_availability', '') in HOURS_OPTIONS else 0
        avail = st.selectbox('Weekly Learning Hours', options=HOURS_OPTIONS, index=avail_idx)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button('Back', use_container_width=True):
                st.session_state.onboarding_step = 2
                st.rerun()
        with c2:
            if st.button('Complete Setup', use_container_width=True, type='primary'):
                profile.update({
                    'ai_learning_goals': ai_goals.strip(),
                    'weekly_learning_availability': avail,
                    'onboarding_completed': True,
                })
                result = save_profile(profile)
                if result:
                    st.session_state.profile = result
                    st.session_state.pop('onboarding_step', None)
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def profile_edit_page() -> None:
    profile = st.session_state.get('profile', {})
    st.markdown("<div class='card card-medium'>", unsafe_allow_html=True)
    st.markdown("<h3>Edit Profile</h3>", unsafe_allow_html=True)

    if st.button('Cancel', use_container_width=True):
        st.session_state.pop('editing_profile', None)
        st.rerun()

    with st.form('profile_edit_form'):
        name = st.text_input('Full Name', value=profile.get('full_name', '') or '')
        industry = st.selectbox('Industry', options=INDUSTRIES, index=0)
        exp_label = _value_to_exp(profile.get('years_experience') or 0)
        exp_idx = EXP_RANGES.index(exp_label) if exp_label in EXP_RANGES else 0
        exp = st.selectbox('Years of Experience', options=EXP_RANGES, index=exp_idx)
        goal = st.text_input('Career Goal', value=profile.get('career_aspirations', '') or '')
        ai_goals = st.text_input('AI Learning Goals', value=profile.get('ai_learning_goals', '') or '')
        avail_idx = HOURS_OPTIONS.index(profile.get('weekly_learning_availability', '')) \
            if profile.get('weekly_learning_availability', '') in HOURS_OPTIONS else 0
        avail = st.selectbox('Weekly Learning Hours', options=HOURS_OPTIONS, index=avail_idx)
        if st.form_submit_button('Save Changes', use_container_width=True, type='primary'):
            payload = {
                'full_name': name.strip(), 'industry': industry,
                'years_experience': _exp_to_value(exp),
                'career_aspirations': goal.strip(),
                'ai_learning_goals': ai_goals.strip(),
                'weekly_learning_availability': avail,
                'onboarding_completed': True,
            }
            result = save_profile(payload)
            if result:
                st.session_state.profile = result
                st.session_state.pop('editing_profile', None)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# Mentor chat
# ---------------------------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault('chat_history', [])
    st.session_state.setdefault('chat_scope_key', None)


def build_scope_key(mode: str, level: str | None = None,
                    module: str | None = None, session: int | None = None) -> str:
    return 'global' if mode == 'global' else f'filtered::{level}::{module}::{session}'


def get_module_label(k: str) -> str:
    return MODULE_LABELS.get(k, k.replace('_', ' ').title())


def get_level_label(k: str) -> str:
    return LEVEL_LABELS.get(k, k.title())


@st.cache_data(show_spinner=False)
def fetch_levels() -> list[str]:
    resp = requests.get(f'{API_BASE_URL}/levels', headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        return [str(x) for x in data.get('levels', [])]
    return []


@st.cache_data(show_spinner=False)
def fetch_modules(level: str | None = None) -> list[str]:
    params = {'level': level} if level else {}
    resp = requests.get(f'{API_BASE_URL}/modules', params=params, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        return [str(x) for x in data.get('modules', [])]
    return []


def chat_request(payload: dict) -> dict:
    resp = requests.post(f'{API_BASE_URL}/chat', json=payload, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        answer = data.get('answer') or data.get('response') or data.get('message')
        if answer and str(answer).strip():
            sources = data.get('sources', [])
            if not isinstance(sources, list):
                sources = []
            return {'answer': answer, 'sources': sources}
    raise ValueError('Invalid API response.')


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return ''
    lines = ['**Relevant sources:**']
    for s in sources:
        cit = s.get('citation', '?')
        lvl = get_level_label(str(s.get('level', '')))
        mod = s.get('module', '?')
        ses = s.get('session', '?')
        chk = s.get('chunk', '?')
        txt = ' '.join(str(s.get('text', '')).split())[:140]
        lines.append(f"- `{cit}` — {lvl} | {mod} | S{ses} | C{chk}" + (f" | {txt}" if txt else ''))
    return '\n'.join(lines)


def disclaimer() -> None:
    st.markdown(
        "<p style='text-align: center; color: #9ca3af; font-size: 0.78rem; padding: 1.5rem 0;'>"
        "This content is shared for educational purposes. "
        "All rights belong to IIT Madras, Futurense, and the respective instructors.</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Navigation sidebar
# ---------------------------------------------------------------------------

def render_nav_sidebar() -> str:
    with st.sidebar:
        p = st.session_state.get('profile', {})
        name = p.get('full_name', '') or st.session_state.get('user_email', '')
        st.markdown(f"**{name}**")
        st.divider()

        page_to_label = {'dashboard': 'Dashboard', 'modules': 'Modules', 'assessment': 'Assessment', 'learning_path': 'Learning Path', 'mentor': 'AI Mentor', 'profile': 'Profile'}
        labels = ['Dashboard', 'Modules', 'Assessment', 'Learning Path', 'AI Mentor', 'Profile']
        current_page = st.session_state.get('page', 'dashboard')
        current_label = page_to_label.get(current_page, 'Dashboard')
        selected = st.radio(
            'Navigate',
            options=labels,
            index=labels.index(current_label),
            label_visibility='collapsed',
        )
        label_to_page = {'Dashboard': 'dashboard', 'Modules': 'modules', 'Assessment': 'assessment', 'Learning Path': 'learning_path', 'AI Mentor': 'mentor', 'Profile': 'profile'}
        st.session_state.page = label_to_page[selected]
        st.query_params['p'] = st.session_state.page

        if st.session_state.page == 'mentor':
            st.divider()
            st.markdown("**Chat Scope**")
            chat_mode = st.radio(
                'Scope',
                options=['All Content', 'Select Level & Module'],
                index=0 if st.session_state.get('chat_mode') != 'Select Level & Module' else 1,
                label_visibility='collapsed',
            )
            st.session_state.chat_mode = chat_mode

            if chat_mode == 'Select Level & Module':
                try:
                    levels = fetch_levels()
                    if levels:
                        sel_level = st.selectbox(
                            'Level', levels, index=None, placeholder='Select',
                            format_func=get_level_label,
                            key='chat_level',
                        )
                        if sel_level:
                            try:
                                mods = fetch_modules(sel_level)
                                if mods:
                                    st.selectbox(
                                        'Module', mods, index=None, placeholder='Select',
                                        format_func=get_module_label,
                                        key='chat_module',
                                    )
                            except requests.RequestException:
                                pass
                except requests.RequestException:
                    pass

        st.divider()
        if st.button('Sign Out', use_container_width=True):
            do_logout()
            st.rerun()

    return st.session_state.page


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

def dashboard_page() -> None:
    p = st.session_state.get('profile', {})
    name = p.get('full_name', '')
    greeting = f', {name}' if name else ''

    assessment = _fetch_latest_assessment()
    assessment_result = assessment.get('result', {}) if assessment else None
    track = assessment_result.get('recommended_track', '') if assessment_result else ''
    track_info = TRACK_INFO.get(track, {})

    roadmap_data = _fetch_active_roadmap()
    roadmap = roadmap_data.get('roadmap', {}) if roadmap_data else None
    weeks = roadmap.get('weeks', []) if roadmap else []

    st.markdown("<h1 class='page-title'>Welcome{greeting}</h1>", unsafe_allow_html=True)

    if track:
        track_cls = f"badge-{track}"
        st.markdown(
            f"<div class='flex-center' style='margin-bottom: 1.5rem;'>"
            f"<span class='text-muted'>Current Track:</span>"
            f"<span class='badge {track_cls}'>{track_info.get('label', track)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p class='text-muted page-subtitle'>Complete the assessment to get your personalized learning track.</p>",
                    unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if assessment_result:
            score = assessment_result.get('total_score', 0)
            track_cls = f"badge-{track}"
            st.markdown(
                f"<div class='card' style='height: 100%;'>"
                f"<div style='font-weight: 700; margin-bottom: 0.75rem;'>📊 Assessment Result</div>"
                f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                f"<div><span class='stat-number' style='color: {track_info.get('color', '#6b7280')};'>{score}</span><span class='stat-label'>/75</span></div>"
                f"<div class='badge {track_cls}'>{track_info.get('label', '')}</div>"
                f"</div>"
                f"<div class='mt-1'>"
                f"<span style='color: #10b981;'>{len(assessment_result.get('strengths', []))} strengths</span>"
                f"<span style='color: #9ca3af; margin: 0 0.5rem;'>·</span>"
                f"<span style='color: #f59e0b;'>{len(assessment_result.get('gaps', []))} growth areas</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='card' style='height: 100%; text-align: center;'>"
                f"<div style='font-size: 2rem; margin-bottom: 0.5rem;'>📝</div>"
                f"<div style='font-weight: 600;'>Take the Assessment</div>"
                f"<div class='text-muted' style='font-size: 0.9rem; margin: 0.5rem 0 1rem;'>Discover your AI readiness level</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button('Start Assessment', key='dash_start_assessment', use_container_width=True, type='primary'):
                st.session_state.page = 'assessment'
                st.query_params['p'] = 'assessment'
                st.rerun()

    with col2:
        if weeks:
            total = len(weeks)
            pct = 0
            st.markdown(
                f"<div class='card' style='height: 100%;'>"
                f"<div style='font-weight: 700; margin-bottom: 0.75rem;'>🗺️ Learning Path Progress</div>"
                f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>"
                f"<div><span class='stat-number'>0</span><span class='stat-label'>/{total} weeks</span></div>"
                f"<div class='text-muted' style='font-size: 0.85rem;'>{roadmap.get('estimated_duration', '4 weeks')}</div>"
                f"</div>"
                f"<div style='background: #e5e7eb; border-radius: 8px; height: 8px;'>"
                f"<div style='background: #6366f1; border-radius: 8px; height: 8px; width: {pct}%;'></div>"
                f"</div>"
                f"<div class='mt-1 text-muted' style='font-size: 0.85rem;'>"
                f"Week 1: {weeks[0].get('focus', '') if weeks else ''}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='card' style='height: 100%; text-align: center;'>"
                f"<div style='font-size: 2rem; margin-bottom: 0.5rem;'>🗺️</div>"
                f"<div style='font-weight: 600;'>No Learning Path Yet</div>"
                f"<div class='text-muted' style='font-size: 0.9rem; margin: 0.5rem 0 1rem;'>Complete the assessment first</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='mt-1'></div>", unsafe_allow_html=True)

    cols = st.columns(5)
    actions = [
        ('📚', 'Modules', 'modules'),
        ('📝', 'Assessment', 'assessment'),
        ('🗺️', 'Learning Path', 'learning_path'),
        ('🤖', 'AI Mentor', 'mentor'),
        ('👤', 'Profile', 'profile'),
    ]
    for col, (icon, label, target) in zip(cols, actions):
        with col:
            if st.button(f'{icon} {label}', key=f'nav_{target}', use_container_width=True):
                st.session_state.page = target
                st.query_params['p'] = target
                st.rerun()


# ---------------------------------------------------------------------------
# Assessment page (placeholder)
# ---------------------------------------------------------------------------

ASSESSMENT_QUESTIONS = [
    {
        'id': 'q1',
        'text': 'What is the most repetitive task you perform daily that you wish a computer could handle?',
        'options': [
            {'label': 'Writing routine emails/reports', 'value': 'A'},
            {'label': 'Analyzing data or customer feedback', 'value': 'B'},
            {'label': 'Building custom tools or automated systems', 'value': 'C'},
        ],
    },
    {
        'id': 'q2',
        'text': 'How much time per week do you believe could be saved if you had an AI partner?',
        'options': [
            {'label': '1-2 hours', 'value': 'A'},
            {'label': '3-5 hours', 'value': 'B'},
            {'label': '5+ hours', 'value': 'C'},
        ],
    },
    {
        'id': 'q3',
        'text': 'Which department function do you think is currently most behind in using AI?',
        'options': [
            {'label': 'General Administration', 'value': 'A'},
            {'label': 'Operations/Project Management', 'value': 'B'},
            {'label': 'IT Infrastructure/Development', 'value': 'C'},
        ],
    },
    {
        'id': 'q4',
        'text': 'How do you currently ask an AI to help you?',
        'options': [
            {'label': 'Casual, simple queries', 'value': 'A'},
            {'label': 'Structured, persona-based prompts', 'value': 'B'},
            {'label': 'Iterative chains of prompts and fine-tuned instructions', 'value': 'C'},
        ],
    },
    {
        'id': 'q5',
        'text': 'What do you do when an AI gives you an almost perfect answer?',
        'options': [
            {'label': 'I re-type the question or move on', 'value': 'A'},
            {'label': 'I provide more context/instructions to refine it', 'value': 'B'},
            {'label': 'I adjust model settings or chain new prompts to fix it', 'value': 'C'},
        ],
    },
    {
        'id': 'q6',
        'text': 'Have you ever tried to chain multiple AI requests to get a complex outcome?',
        'options': [
            {'label': 'No', 'value': 'A'},
            {'label': 'Occasionally', 'value': 'B'},
            {'label': 'Frequently/As part of my workflow', 'value': 'C'},
        ],
    },
    {
        'id': 'q7',
        'text': 'How comfortable are you with terms like API, Model, or RAG?',
        'options': [
            {'label': '1-2 (Not comfortable)', 'value': 'A'},
            {'label': '3 (Somewhat comfortable)', 'value': 'B'},
            {'label': '4-5 (Very comfortable)', 'value': 'C'},
        ],
    },
    {
        'id': 'q8',
        'text': 'Do you currently use AI features inside apps (e.g., Copilot in Excel, CRM AI)?',
        'options': [
            {'label': 'No, I stick to basic chat', 'value': 'A'},
            {'label': 'Yes, occasionally', 'value': 'B'},
            {'label': 'Yes, I actively leverage them for deep tasks', 'value': 'C'},
        ],
    },
    {
        'id': 'q9',
        'text': 'Are you interested in learning how to connect AI to your own data or spreadsheets?',
        'options': [
            {'label': 'Perhaps later', 'value': 'A'},
            {'label': 'Yes, definitely', 'value': 'B'},
            {'label': 'Yes, it is a priority', 'value': 'C'},
        ],
    },
    {
        'id': 'q10',
        'text': 'What is your primary concern when sharing company information with an AI?',
        'options': [
            {'label': 'Not saying anything wrong/biased', 'value': 'A'},
            {'label': 'Data privacy/company policy', 'value': 'B'},
            {'label': 'Scalability and system vulnerabilities', 'value': 'C'},
        ],
    },
    {
        'id': 'q11',
        'text': 'How do you verify if the information provided by an AI is actually accurate?',
        'options': [
            {'label': 'I trust the first result', 'value': 'A'},
            {'label': 'I do a quick manual check', 'value': 'B'},
            {'label': 'I perform cross-verification and logic testing', 'value': 'C'},
        ],
    },
    {
        'id': 'q12',
        'text': 'Are you familiar with your company internal guidelines for using AI safely?',
        'options': [
            {'label': 'No', 'value': 'A'},
            {'label': 'Partially', 'value': 'B'},
            {'label': 'Yes, fully familiar', 'value': 'C'},
        ],
    },
    {
        'id': 'q13',
        'text': 'What is your dream outcome from this training?',
        'options': [
            {'label': 'Save time on daily tasks', 'value': 'A'},
            {'label': 'Lead AI projects in my department', 'value': 'B'},
            {'label': 'Build/deploy AI solutions for the team', 'value': 'C'},
        ],
    },
    {
        'id': 'q14',
        'text': 'Do you prefer learning by reading theory or by building actual AI projects?',
        'options': [
            {'label': 'Theory/Articles', 'value': 'A'},
            {'label': 'Guided walkthroughs', 'value': 'B'},
            {'label': 'Hands-on project building', 'value': 'C'},
        ],
    },
    {
        'id': 'q15',
        'text': 'Are you aiming to be a user of AI tools or an architect of AI solutions?',
        'options': [
            {'label': 'User', 'value': 'A'},
            {'label': 'Integrator', 'value': 'B'},
            {'label': 'Architect', 'value': 'C'},
        ],
    },
]

TRACK_INFO = {
    'foundations': {
        'label': 'AI Foundations',
        'color': '#10b981',
        'description': 'You are in the early stages of your AI journey. This track will build your foundational understanding of AI concepts, prompt engineering, and ethical AI use.',
    },
    'practitioner': {
        'label': 'AI Practitioner',
        'color': '#f59e0b',
        'description': 'You have solid AI awareness and are ready to integrate AI into your workflows. This track deepens your skills in data analysis, RAG, and human-in-the-loop systems.',
    },
    'builder': {
        'label': 'AI Builder',
        'color': '#ef4444',
        'description': 'You are technically advanced and ready to architect AI solutions. This track covers enterprise AI strategy, agentic frameworks, and model optimization.',
    },
}


def _submit_assessment(answers: list[dict]) -> dict | None:
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = requests.post(
            f'{API_BASE_URL}/v1/assessment/submit',
            json={'answers': answers},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f'Error {resp.status_code}: {resp.text[:200]}')
    except requests.ConnectionError:
        st.error('Cannot reach the backend.')
    except requests.Timeout:
        st.error('Request timed out.')
    except requests.RequestException as exc:
        st.error(str(exc))
    return None


def _fetch_latest_assessment() -> dict | None:
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = requests.get(
            f'{API_BASE_URL}/v1/assessment/result',
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def _fetch_track_modules(track_id: str) -> list | None:
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = requests.get(
            f'{API_BASE_URL}/v1/tracks/{track_id}/modules',
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get('modules', [])
    except requests.RequestException:
        pass
    return None


def _generate_roadmap(track_id: str) -> dict | None:
    headers = _auth_headers()
    if not headers:
        return None
    profile = st.session_state.get('profile', {})
    try:
        resp = requests.post(
            f'{API_BASE_URL}/v1/roadmap/generate',
            json={
                'track_id': track_id,
                'industry': profile.get('industry'),
                'job_role': profile.get('job_role'),
            },
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f'Error {resp.status_code}: {resp.text[:200]}')
    except requests.ConnectionError:
        st.error('Cannot reach the backend.')
    except requests.RequestException as exc:
        st.error(str(exc))
    return None


def _fetch_active_roadmap() -> dict | None:
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = requests.get(
            f'{API_BASE_URL}/v1/roadmap/active',
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def _render_assessment_result(result_data: dict) -> None:
    result = result_data.get('result', {})
    track = result.get('recommended_track', 'foundations')
    track_info = TRACK_INFO.get(track, TRACK_INFO['foundations'])
    total_score = result.get('total_score', 0)

    track_cls = f"badge-{track}"
    st.markdown(
        f"<div style='text-align: center; padding: 2rem 1rem;'>"
        f"<h2 style='margin-bottom: 0.5rem;'>Your AI Readiness Result</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='card' style='max-width: 700px; text-align: center;'>"
        f"<div class='text-muted' style='margin-bottom: 0.5rem;'>Your Score</div>"
        f"<div class='score-number' style='color: {track_info['color']};'>{total_score}/75</div>"
        f"<div style='font-size: 1.5rem; font-weight: 700; color: {track_info['color']}; margin: 1rem 0 0.5rem;'>{track_info['label']}</div>"
        f"<div style='color: #374151; line-height: 1.6;'>{track_info['description']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    category_scores = result.get('category_scores', [])
    if category_scores:
        st.markdown("<div class='card' style='max-width: 700px;'>", unsafe_allow_html=True)
        st.markdown("<h3>Category Breakdown</h3>", unsafe_allow_html=True)
        for cat in category_scores:
            label = cat.get('label', cat.get('category', ''))
            score = cat.get('score', 0)
            pct = int((score / 5) * 100)
            bar_color = '#10b981' if pct >= 70 else '#f59e0b' if pct >= 50 else '#ef4444'
            st.markdown(
                f"<div style='margin-bottom: 0.75rem;'>"
                f"<div style='display: flex; justify-content: space-between;'>"
                f"<span>{label}</span><span>{score}/5</span>"
                f"</div>"
                f"<div style='background: #e5e7eb; border-radius: 8px; height: 10px;'>"
                f"<div style='background: {bar_color}; border-radius: 8px; height: 10px; width: {pct}%; transition: width 0.3s ease;'></div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    strengths = result.get('strengths', [])
    gaps = result.get('gaps', [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #10b981;'>Strengths</h3>", unsafe_allow_html=True)
        for s in strengths:
            st.markdown(f"✅ {s}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #f59e0b;'>Growth Areas</h3>", unsafe_allow_html=True)
        for g in gaps:
            st.markdown(f"📈 {g}")
        st.markdown("</div>", unsafe_allow_html=True)

    track_modules = _fetch_track_modules(track)
    if track_modules:
        st.markdown("<div class='card' style='max-width: 700px;'>", unsafe_allow_html=True)
        st.markdown("<h3>Recommended Learning Path</h3>", unsafe_allow_html=True)
        for mod in track_modules:
            mod_name = mod.get('module', '').replace('_', ' ').title()
            sessions = mod.get('sessions', [])
            session_count = len(sessions)
            role_tag = ''
            if mod.get('role_specific'):
                role_tag = ' <span style="background: #e0f2fe; color: #0369a1; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.5rem;">Role-specific</span>'
            st.markdown(
                f"<div style='padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;'>"
                f"<span style='font-weight: 600;'>{mod_name}</span>{role_tag}"
                f"<span style='float: right; color: #6b7280; font-size: 0.85rem;'>{session_count} session{'s' if session_count != 1 else ''}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button('Generate My Learning Path', use_container_width=True, type='primary'):
            st.session_state.page = 'learning_path'
            st.query_params['p'] = 'learning_path'
            st.rerun()
    with c2:
        if st.button('Retake Assessment', use_container_width=True):
            st.session_state.assessment_done = False
            st.session_state.assessment_result = None
            st.rerun()


def assessment_page() -> None:
    st.markdown("<h1 class='page-title'>AI Readiness Assessment</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-subtitle'>"
        "Discover your AI skill level and get a personalized learning track recommendation.</p>",
        unsafe_allow_html=True,
    )

    if 'assessment_done' not in st.session_state:
        st.session_state.assessment_done = False
    if 'assessment_result' not in st.session_state:
        st.session_state.assessment_result = None

    if st.session_state.assessment_result is None:
        existing = _fetch_latest_assessment()
        if existing:
            st.session_state.assessment_result = existing

    if st.session_state.assessment_result and st.session_state.assessment_done is False:
        st.session_state.assessment_done = True

    if st.session_state.assessment_done and st.session_state.assessment_result:
        _render_assessment_result(st.session_state.assessment_result)
        return

    if not st.session_state.assessment_done:
        st.markdown(
            "<div class='card' style='max-width: 700px;'>"
            "<p style='color: #374151; line-height: 1.7;'>"
            "This assessment includes 15 questions covering your AI experience, workflow habits, "
            "and learning preferences. Based on your answers, you'll be mapped to one of three tracks: "
            "<strong>Foundations</strong>, <strong>Intermediate</strong>, or <strong>Advanced</strong>.</p>"
            "<p style='color: #374151; line-height: 1.7;'>"
            "The assessment takes about 5 minutes. You can retake it anytime.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button('Start Assessment', use_container_width=True, type='primary'):
            st.session_state.assessment_done = None
            st.rerun()

    if st.session_state.assessment_done is None:
        with st.form('assessment_form'):
            answers = {}
            for i, q in enumerate(ASSESSMENT_QUESTIONS, 1):
                st.markdown(f"**Q{i}.** {q['text']}")
                opts = {o['label']: o['value'] for o in q['options']}
                selected = st.radio(
                    '',
                    list(opts.keys()),
                    index=None,
                    key=f'assess_{q["id"]}',
                    label_visibility='collapsed',
                )
                if selected:
                    answers[q['id']] = opts[selected]
                st.markdown("<hr style='margin: 0.75rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

            submitted = st.form_submit_button('Submit Assessment', use_container_width=True, type='primary')
            if submitted:
                if len(answers) < len(ASSESSMENT_QUESTIONS):
                    st.error(f'Please answer all {len(ASSESSMENT_QUESTIONS)} questions before submitting.')
                else:
                    formatted = [{'question_id': qid, 'selected_value': val} for qid, val in answers.items()]
                    with st.spinner('Scoring your assessment...'):
                        result = _submit_assessment(formatted)
                    if result:
                        st.session_state.assessment_done = True
                        st.session_state.assessment_result = result
                        st.rerun()


# ---------------------------------------------------------------------------
# Learning Path page
# ---------------------------------------------------------------------------

def learning_path_page() -> None:
    st.markdown("<h1 class='page-title'>Your Learning Path</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-subtitle'>A personalized 4-week study plan based on your assessment and profile.</p>",
        unsafe_allow_html=True,
    )

    if 'roadmap_data' not in st.session_state:
        st.session_state.roadmap_data = None

    if st.session_state.roadmap_data is None:
        existing = _fetch_active_roadmap()
        if existing:
            st.session_state.roadmap_data = existing

    if st.session_state.roadmap_data is None:
        latest_assessment = _fetch_latest_assessment()
        if not latest_assessment:
            st.warning('Complete the AI Readiness Assessment first to generate your learning path.')
            if st.button('Go to Assessment', use_container_width=True, type='primary'):
                st.session_state.page = 'assessment'
                st.query_params['p'] = 'assessment'
                st.rerun()
            return

        track = latest_assessment.get('result', {}).get('recommended_track', 'foundations')
        if st.button('Generate My 4-Week Learning Plan', use_container_width=True, type='primary'):
            with st.spinner('Creating your personalized learning path...'):
                result = _generate_roadmap(track)
                if result:
                    st.session_state.roadmap_data = result
                    st.rerun()
        return

    roadmap = st.session_state.roadmap_data.get('roadmap', {})
    track_info = roadmap.get('track', {})
    weeks = roadmap.get('weeks', [])

    st.markdown(
        f"<div class='card' style='max-width: 800px;'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
        f"<div><strong style='font-size: 1.2rem;'>{track_info.get('label', '')}</strong>"
        f"<br><span style='color: #6b7280;'>{roadmap.get('estimated_duration', '')} · ~{roadmap.get('estimated_total_hours', 0)} hours total</span></div>"
        f"<div style='background: #f3f4f6; padding: 0.5rem 1rem; border-radius: 8px; text-align: center;'>"
        f"<div style='font-size: 1.5rem; font-weight: 700;'>{len(weeks)}</div>"
        f"<div style='font-size: 0.8rem; color: #6b7280;'>Weeks</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    for week in weeks:
        week_num = week.get('week', 0)
        focus = week.get('focus', '')
        hours = week.get('estimated_hours', 0)
        objectives = week.get('objectives', [])
        sessions_detail = week.get('sessions_detail', [])

        is_expanded = week_num == 1
        expander_label = f'Week {week_num}: {focus} ({hours}h)'

        with st.expander(expander_label, expanded=is_expanded):
            st.markdown("**Learning Objectives**")
            for obj in objectives:
                st.markdown(f"- {obj}")

            if sessions_detail:
                st.markdown("---")
                st.markdown("**Recommended Sessions**")
                for sd in sessions_detail:
                    label = sd.get('label', sd.get('module', ''))
                    snums = sd.get('session_numbers', [])
                    avail = sd.get('available_sessions', [])
                    role_tag = ''
                    if sd.get('role_specific'):
                        role_tag = ' 🎯 Role-specific'
                    sessions_str = ', '.join(str(s) for s in snums)
                    st.markdown(f"- **{label}**{role_tag} — Session{'s' if len(snums) != 1 else ''} {sessions_str}")
                    if avail:
                        st.caption(f"  Available sessions: {', '.join(str(s) for s in avail)}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Start AI Mentor', use_container_width=True, type='primary'):
            st.session_state.page = 'mentor'
            st.query_params['p'] = 'mentor'
            st.rerun()
    with col2:
        if st.button('Regenerate Plan', use_container_width=True):
            with st.spinner('Regenerating...'):
                track = track_info.get('id', 'foundations')
                result = _generate_roadmap(track)
                if result:
                    st.session_state.roadmap_data = result
                    st.rerun()


# ---------------------------------------------------------------------------
# Profile page
# ---------------------------------------------------------------------------

def profile_page() -> None:
    profile = st.session_state.get('profile', {})
    st.markdown("<h1 class='page-title'>Your Profile</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-subtitle'>Manage your learning preferences.</p>",
        unsafe_allow_html=True,
    )

    if st.button('Edit Profile', use_container_width=True):
        st.session_state.editing_profile = True
        st.rerun()

    if st.session_state.get('editing_profile'):
        profile_edit_page()
        return

    st.markdown("<div class='card' style='max-width: 700px;'>", unsafe_allow_html=True)
    fields = [
        ('Full Name', profile.get('full_name', '—')),
        ('Industry', profile.get('industry', '—')),
        ('Years of Experience', str(profile.get('years_experience', '—'))),
        ('Career Goal', profile.get('career_aspirations', '—')),
        ('AI Learning Goals', profile.get('ai_learning_goals', '—')),
        ('Weekly Availability', profile.get('weekly_learning_availability', '—')),
    ]
    for label, value in fields:
        st.markdown(f"**{label}:** {value}")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Mentor / Chat page
# ---------------------------------------------------------------------------

def mentor_page() -> None:
    init_state()

    st.markdown("<h1 class='page-title'>AI Mentor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Ask me anything about your course.</p>",
                unsafe_allow_html=True)

    with st.expander('About AI Mentor', expanded=True):
        st.markdown("""
**AI Mentor** — Clear your doubts. Learn with confidence.

**Who is it for?**  
- Learners at Level 1, Level 2, and Level 3  
- Students who want quick clarification on complex topics  
- Anyone who wants support while studying AI and related modules

**What problem it solves**  
- Removes confusion when concepts feel unclear  
- Helps you understand topics without getting stuck  
- Gives guided support based on your learning stage

**How it helps**  
- Answers your questions in a clear and simple way  
- Explains topics using the course content and learning stages  
- Supports deeper understanding with follow-up questions  
- Makes learning feel more structured, interactive, and confident
        """)
        st.caption('Ask one question at a time. The chatbot remembers the last 5 exchanges.')

    chat_mode = st.session_state.get('chat_mode', 'All Content')
    level = st.session_state.get('chat_level') if chat_mode == 'Select Level & Module' else None
    module = st.session_state.get('chat_module') if chat_mode == 'Select Level & Module' else None

    mode_param = 'global' if chat_mode == 'All Content' else 'filtered'
    scope_key = build_scope_key(mode_param, level, module, None)
    messages = ensure_history(scope_key)

    for msg in messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    prompt = st.chat_input('Ask a question')
    if not prompt:
        disclaimer()
        return

    messages.append({'role': 'user', 'content': prompt})
    st.session_state.chat_history = trim_chat_history(messages)
    messages = st.session_state.chat_history
    with st.chat_message('user'):
        st.markdown(prompt)

    payload = {'question': prompt, 'mode': mode_param, 'chat_history': messages}
    if mode_param != 'global':
        payload.update({'level': level, 'module': module})

    try:
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                data = chat_request(payload)
                rendered = data['answer']
                src = format_sources(data['sources'])
                if src:
                    rendered += '\n\n' + src
                st.markdown(rendered)
    except (requests.RequestException, ValueError) as exc:
        msg = str(exc)
        with st.chat_message('assistant'):
            st.error(msg)
        messages.append({'role': 'assistant', 'content': msg})
        st.session_state.chat_history = trim_chat_history(messages)
        disclaimer()
        return

    messages.append({'role': 'assistant', 'content': rendered})
    st.session_state.chat_history = trim_chat_history(messages)
    disclaimer()


def trim_chat_history(h: list) -> list:
    return h[-MAX_HISTORY_MESSAGES:]


def ensure_history(key: str) -> list:
    if st.session_state.chat_scope_key != key:
        st.session_state.chat_scope_key = key
        st.session_state.chat_history = []
    return st.session_state.chat_history


# ---------------------------------------------------------------------------
# Modules / Content Catalog page
# ---------------------------------------------------------------------------

def modules_page() -> None:
    st.markdown("<h1 class='page-title'>Content Catalog</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-subtitle'>"
        "Browse all available modules organized by level. Each module contains lecture sessions "
        "you can explore through the AI Mentor.</p>",
        unsafe_allow_html=True,
    )

    levels = {
        'beginner': ('Foundations', 1, '#10b981'),
        'intermediate': ('Intermediate', 2, '#f59e0b'),
        'advanced': ('Advanced', 3, '#ef4444'),
    }

    categories = {
        'beginner': [
            ('Common Modules', ['ai_foundations_curriculum', 'prompt_engineering', 'ai_ethics_safety_and_data_privacy']),
            ('Role-Specific', ['finance_chatgpt_excel_skills', 'hr_ai_enhanced_jd_design_and_skills_gap_mapping', 'operations_process_mapping_and_automated_reporting']),
        ],
        'intermediate': [
            ('Common Modules', ['ai_data_analysis_extracting_insights', 'human_in_the_loop_designing_hybrid_systems']),
            ('Role-Specific', ['customer_facing_ai_sentiment_analysis_and_crm_integration', 'project_management_predictive_resource_allocation_and_automated_risk_tracking']),
        ],
        'advanced': [
            ('Core', ['cms', 'map', 'wdp']),
        ],
    }

    for level_key in ('beginner', 'intermediate', 'advanced'):
        level_label, level_order, level_color = levels[level_key]
        with st.expander(f"**{level_label}**", icon='📘' if level_key == 'beginner' else ('📙' if level_key == 'intermediate' else '📕'), expanded=level_key == 'beginner'):
            for cat_name, module_ids in categories[level_key]:
                st.markdown(f"<p class='text-muted' style='font-size: 0.9rem; margin: 0.5rem 0 0.25rem;'><strong>{cat_name}</strong></p>", unsafe_allow_html=True)
                cols = st.columns(2)
                for i, mod_id in enumerate(module_ids):
                    with cols[i % 2]:
                        label = MODULE_LABELS.get(mod_id, mod_id.replace('_', ' ').title())
                        desc = MODULE_DESCRIPTIONS.get(mod_id, '')
                        sessions = MODULE_SESSION_COUNTS.get(mod_id, '?')
                        track_id = MODULE_TRACK_MAP.get(mod_id, '')
                        track_label = TRACK_INFO.get(track_id, {}).get('label', '')
                        track_color = TRACK_INFO.get(track_id, {}).get('color', '#6b7280')

                        track_cls = f"badge-{track_id}" if track_id else ''
                        session_label = f'{sessions} session{"s" if sessions != 1 else ""}'
                        st.markdown(
                            f"<div class='card-sm card'>"
                            f"<div style='font-weight: 600; font-size: 0.95rem;'>{label}</div>"
                            f"<div class='text-muted' style='font-size: 0.82rem; margin: 0.25rem 0;'>{desc}</div>"
                            f"<div style='display: flex; gap: 0.5rem; align-items: center; margin-top: 0.35rem;'>"
                            f"<span class='text-light' style='font-size: 0.75rem;'>{session_label}</span>"
                            f"<span style='font-size: 0.7rem; color: #9ca3af;'>|</span>"
                            f"<span class='badge-pill {track_cls}'>{track_label}</span>"
                            f"</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[1]:
        if st.button('🤖 Ask the AI Mentor', use_container_width=True, type='primary'):
            st.session_state.page = 'mentor'
            st.query_params['p'] = 'mentor'
            st.rerun()


# ---------------------------------------------------------------------------
# App shell — sidebar nav + page routing
# ---------------------------------------------------------------------------

def app_shell() -> None:
    inject_css()
    current_page = render_nav_sidebar()

    page_map = {
        'dashboard': dashboard_page,
        'modules': modules_page,
        'assessment': assessment_page,
        'learning_path': learning_path_page,
        'mentor': mentor_page,
        'profile': profile_page,
    }
    page_map.get(current_page, dashboard_page)()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

st.set_page_config(page_title='AI Mentor', page_icon='🤖', layout='wide')

_recover_session()

if not is_authenticated():
    inject_css()
    auth_page()

if st.session_state.get('profile') is None:
    p = fetch_profile()
    st.session_state.profile = p or {}

if not onboarding_done():
    onboarding_page()

if st.session_state.get('page') is None:
    st.session_state.page = st.query_params.get('p') or 'dashboard'

app_shell()

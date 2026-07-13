from __future__ import annotations

from typing import Any

import requests
import streamlit as st
from supabase import create_client

from config import get_env


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
    'cms': 'Contextual Reasoning for Multi-Agent Systems',
    'map': 'Multi-Agent Planning & Workflow Design',
    'wdp': 'Workflow Design & Optimization',
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


def _verify_token_with_backend() -> dict | None:
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = requests.get(f'{API_BASE_URL}/v1/auth/me', headers=headers, timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except requests.RequestException:
        return None


def do_login(email: str, password: str) -> str | None:
    try:
        client = _supabase_client()
        resp = client.auth.sign_in_with_password({'email': email, 'password': password})
        if resp.session:
            st.session_state.auth_token = resp.session.access_token
            st.session_state.user_email = email
            return None
        return 'Login failed.'
    except Exception as exc:
        return str(exc)


def do_logout() -> None:
    for key in ('auth_token', 'user_email', 'user_info', 'profile',
                'chat_history', 'chat_scope_key'):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return bool(st.session_state.get('auth_token'))


# ---------------------------------------------------------------------------
# Profile / Onboarding helpers
# ---------------------------------------------------------------------------

def fetch_profile() -> dict | None:
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = requests.get(f'{API_BASE_URL}/v1/profile', headers=headers, timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except requests.RequestException:
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
        .stApp { background: #fafafa; }
        .block-container { max-width: 1100px; padding-top: 0.5rem; }
        .auth-card {
            max-width: 420px; margin: 0 auto; padding: 2.5rem 2rem;
            background: #fff; border-radius: 14px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
        }
        .auth-card h2 { font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 0.25rem; }
        .auth-card .subtitle { color: #6b7280; font-size: 0.92rem; margin-bottom: 1.5rem; }
        .auth-card .powered { text-align: center; color: #9ca3af; font-size: 0.78rem; margin-top: 1.5rem; }
        .auth-card .stButton > button { border-radius: 8px; font-weight: 600; }
        .card {
            background: #fff; border-radius: 12px; padding: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem;
        }
        .card-medium { max-width: 700px; margin: 1rem auto; }
        h1, h2, h3 { color: #111827; }
        .stButton > button { border-radius: 8px; font-weight: 600; }
        div[data-testid="stSidebar"] { background: #fff; }
        div[data-testid="stSidebar"] .stButton > button { width: 100%; }
        hr { margin: 2rem 0; }
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
        avail_idx = HOURS_OPTIONS.index(profile['weekly_learning_availability']) \
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


def sidebar() -> tuple[str, str | None, str | None, int | None]:
    with st.sidebar:
        p = st.session_state.get('profile', {})
        name = p.get('full_name', '') or st.session_state.get('user_email', '')
        st.markdown(f"**{name}**")
        if st.button('Edit Profile', use_container_width=True):
            st.session_state.editing_profile = True
            st.rerun()
        if st.button('Sign Out', use_container_width=True):
            do_logout()
            st.rerun()
        st.divider()
        st.markdown("**Chat Scope**")
        mode = st.radio('Mode', ['All Content', 'Select Level & Module'], index=0, label_visibility='collapsed')
        if mode == 'All Content':
            return 'global', None, None, None

        try:
            levels = fetch_levels()
        except requests.RequestException:
            return 'filtered', None, None, None
        if not levels:
            return 'filtered', None, None, None
        sl = st.selectbox('Level', levels, index=None, placeholder='Select', format_func=get_level_label)
        if not sl:
            return 'filtered', None, None, None
        try:
            mods = fetch_modules(sl)
        except requests.RequestException:
            return 'filtered', sl, None, None
        if not mods:
            return 'filtered', sl, None, None
        sm = st.selectbox('Module', mods, index=None, placeholder='Select', format_func=get_module_label)
        return ('filtered', sl, sm, None) if sm else ('filtered', sl, None, None)


def disclaimer() -> None:
    st.markdown(
        "<p style='text-align: center; color: #9ca3af; font-size: 0.78rem; padding: 1.5rem 0;'>"
        "This content is shared for educational purposes. "
        "All rights belong to IIT Madras, Futurense, and the respective instructors.</p>",
        unsafe_allow_html=True,
    )


def mentor_page() -> None:
    if _verify_token_with_backend() is None:
        do_logout()
        st.warning('Session expired.')
        auth_page()
        return

    if st.session_state.get('editing_profile'):
        profile_edit_page()
        return

    init_state()
    inject_css()

    p = st.session_state.get('profile', {})
    name = p.get('full_name', '')
    greeting = f', {name}' if name else ''

    st.markdown(f"<h1 style='margin-bottom: 0.25rem;'>Welcome{greeting}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; margin-bottom: 1.5rem;'>Ask me anything about your course.</p>",
                unsafe_allow_html=True)

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

    mode, level, module, session = sidebar()

    if mode == 'global':
        st.info('Welcome! Ask me anything about your course material.', icon='💬')

    if mode == 'filtered' and (not level or not module):
        st.info('Select a level and module from the sidebar.')
        return

    scope_key = build_scope_key(mode, level, module, session)
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

    payload = {'question': prompt, 'mode': mode, 'chat_history': messages}
    if mode != 'global':
        payload.update({'level': level, 'module': module, 'session': session})

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
# Entry point
# ---------------------------------------------------------------------------

st.set_page_config(page_title='AI Mentor', layout='wide')

if not is_authenticated():
    inject_css()
    auth_page()

if st.session_state.get('profile') is None:
    p = fetch_profile()
    st.session_state.profile = p or {}

if not onboarding_done():
    onboarding_page()

mentor_page()

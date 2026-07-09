from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from config import get_env


API_BASE_URL = get_env(
    'API_BASE_URL',
    'https://iitm-curriculem-intelligence-layer.onrender.com',
)
REQUEST_TIMEOUT = 60
MAX_HISTORY_MESSAGES = 10
PAGE_TITLE = 'IITM Curriculum - AI Mentor (Prototype)'
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


def init_state() -> None:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_scope_key' not in st.session_state:
        st.session_state.chat_scope_key = None


def build_scope_key(
    mode: str,
    level: str | None = None,
    module: str | None = None,
    session: int | None = None,
) -> str:
    if mode == 'global':
        return 'global'
    return f'filtered::{level}::{module}::{session}'


def get_module_label(module_key: str) -> str:
    if module_key in MODULE_LABELS:
        return MODULE_LABELS[module_key]
    return module_key.replace('_', ' ').title()


def get_level_label(level_key: str) -> str:
    return LEVEL_LABELS.get(level_key, level_key.title())


@st.cache_data(show_spinner=False)
def fetch_levels() -> list[str]:
    response = requests.get(f'{API_BASE_URL}/levels', timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        levels = data.get('levels', [])
        return [str(item) for item in levels]
    return []


@st.cache_data(show_spinner=False)
def fetch_modules(level: str | None = None) -> list[str]:
    params: dict[str, str] = {}
    if level:
        params['level'] = level
    response = requests.get(f'{API_BASE_URL}/modules', params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        modules = data.get('modules', [])
        return [str(item) for item in modules]
    return []


def send_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f'{API_BASE_URL}/chat',
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    if isinstance(data, dict):
        answer = data.get('answer') or data.get('response') or data.get('message')
        if isinstance(answer, str) and answer.strip():
            sources = data.get('sources', [])
            if not isinstance(sources, list):
                sources = []
            return {
                'answer': answer,
                'sources': sources,
            }

    raise ValueError('The API response did not include an answer field.')


def format_sources_markdown(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ''

    lines = ['Relevant sources:']
    for source in sources:
        citation = source.get('citation') or 'UNKNOWN'
        level = source.get('level') or 'unknown'
        module = source.get('module') or 'unknown'
        session = source.get('session')
        chunk = source.get('chunk')
        preview_text = ' '.join(str(source.get('text', '')).split())
        if len(preview_text) > 140:
            preview_text = f'{preview_text[:137].rstrip()}...'

        source_line = (
            f"- `{citation}` | level `{get_level_label(str(level))}` | module `{module}` | session `{session}` | chunk `{chunk}`"
        )
        if preview_text:
            source_line = f'{source_line} | {preview_text}'
        lines.append(source_line)
    return '\n'.join(lines)


def trim_chat_history(chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
    return chat_history[-MAX_HISTORY_MESSAGES:]


def ensure_history(scope_key: str) -> list[dict[str, str]]:
    if st.session_state.chat_scope_key != scope_key:
        st.session_state.chat_scope_key = scope_key
        st.session_state.chat_history = []
    return st.session_state.chat_history


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 960px;
        }
        .hero-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            text-align: center;
            color: #4b5563;
            font-size: 1.05rem;
            margin-bottom: 1.25rem;
        }
        .hero-description {
            color: #1f2937;
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }
        .hero-footer {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-top: 1rem;
        }
        .hero-disclaimer {
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.5;
            text-align: center;
            margin-top: 2rem;
        }
        .welcome-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.25rem 1rem;
            background: #fafafa;
            margin-bottom: 1rem;
        }
        .info-bar {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            background: #f8fafc;
            color: #334155;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .chat-shell {
            border: 1px solid #eceff3;
            border-radius: 14px;
            padding: 0.5rem 0.75rem 0.75rem 0.75rem;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(f"<div class='hero-title'>{PAGE_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>AI Mentor across foundations, intermediate, and advanced learning modules</div>",
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(
            """
            <div class='hero-description'>
                <strong>AI Mentor</strong><br>
                Clear your doubts. Learn with confidence.
            </div>
            <div class='hero-description'>
                <strong>Who is it for?</strong><br>
                • Learners at Level 1, Level 2, and Level 3<br>
                • Students who want quick clarification on complex topics<br>
                • Anyone who wants support while studying AI and related modules
            </div>
            <div class='hero-description'>
                <strong>What problem it solves</strong><br>
                • Removes confusion when concepts feel unclear<br>
                • Helps you understand topics without getting stuck<br>
                • Gives guided support based on your learning stage
            </div>
            <div class='hero-description'>
                <strong>How it helps</strong><br>
                • Answers your questions in a clear and simple way<br>
                • Explains topics using the course content and learning stages<br>
                • Supports deeper understanding with follow-up questions<br>
                • Makes learning feel more structured, interactive, and confident
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption('Ask one question at a time for the best experience. The chatbot remembers the last 5 exchanges.')
        st.markdown('---')


def render_history(messages: list[dict[str, str]]) -> None:
    for message in messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            st.write('')


def render_sidebar() -> tuple[str, str | None, str | None, int | None]:
    with st.sidebar:
        st.header('Chat Scope')
        chat_mode_label = st.radio(
            'Chat Mode',
            options=['All Content', 'Select Level & Module'],
            index=0,
        )

        if chat_mode_label == 'All Content':
            return 'global', None, None, None

        try:
            levels = fetch_levels()
        except requests.RequestException as exc:
            st.error(f'Failed to load levels: {exc}')
            return 'filtered', None, None, None

        if not levels:
            st.warning('No levels available.')
            return 'filtered', None, None, None

        selected_level = st.selectbox(
            'Level',
            options=levels,
            index=None,
            placeholder='Select your level',
            format_func=get_level_label,
        )

        if not selected_level:
            return 'filtered', None, None, None

        try:
            modules = fetch_modules(selected_level)
        except requests.RequestException as exc:
            st.error(f'Failed to load modules: {exc}')
            return 'filtered', selected_level, None, None

        if not modules:
            st.warning('No modules available for the selected level.')
            return 'filtered', selected_level, None, None

        selected_module = st.selectbox(
            'Module',
            options=modules,
            index=None,
            placeholder='Select your module',
            format_func=get_module_label,
        )

        if not selected_module:
            return 'filtered', selected_level, None, None

        return 'filtered', selected_level, selected_module, None


def build_payload(
    question: str,
    mode: str,
    level: str | None,
    module: str | None,
    session: int | None,
    chat_history: list[dict[str, str]],
) -> dict[str, Any]:
    if mode == 'global':
        return {
            'question': question,
            'mode': 'global',
            'chat_history': chat_history,
        }

    return {
        'question': question,
        'mode': 'filtered',
        'level': level,
        'module': module,
        'session': session,
        'chat_history': chat_history,
    }


def render_global_welcome() -> None:
    st.markdown(
        """
        <div class="welcome-card">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_scope_summary(
    mode: str,
    level: str | None,
    module: str | None,
    session: int | None,
) -> None:
    if mode != 'filtered' or not level or not module:
        return

    parts = [f'Level: {get_level_label(level)}', f'Module: {get_module_label(module)}']
    if session is not None:
        parts.append(f'Session: {session}')
    st.markdown(
        f"<div class='info-bar'>{' | '.join(parts)}</div>",
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown('---')
    st.markdown(
        """
        <div class='hero-disclaimer'>
            This content is not owned by me and is shared only for educational purposes. All rights belong to IIT Madras, Futurense, and the respective instructors.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout='wide')
    init_state()

    render_styles()
    render_header()

    mode, level, module, session = render_sidebar()
    render_scope_summary(mode, level, module, session)

    if mode == 'global':
        render_global_welcome()

    if mode == 'filtered' and (not level or not module):
        st.info('Select a level and module from the sidebar to start chatting within that scope.')
        return

    scope_key = build_scope_key(mode, level, module, session)
    messages = ensure_history(scope_key)

    with st.container():
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        render_history(messages)
        st.markdown('</div>', unsafe_allow_html=True)

    prompt = st.chat_input('Ask a question')
    if not prompt:
        render_disclaimer()
        return

    messages.append({'role': 'user', 'content': prompt})
    st.session_state.chat_history = trim_chat_history(messages)
    messages = st.session_state.chat_history
    with st.chat_message('user'):
        st.markdown(prompt)
        st.write('')

    payload = build_payload(prompt, mode, level, module, session, messages)

    try:
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                response_data = send_chat_request(payload)
                answer = response_data['answer']
                sources_markdown = format_sources_markdown(response_data['sources'])
                rendered_answer = answer
                if sources_markdown:
                    rendered_answer = f'{answer}\n\n{sources_markdown}'
                st.markdown(rendered_answer)
                st.write('')
    except requests.RequestException as exc:
        error_message = f'Request failed: {exc}'
        with st.chat_message('assistant'):
            st.error(error_message)
        messages.append({'role': 'assistant', 'content': error_message})
        st.session_state.chat_history = trim_chat_history(messages)
        render_disclaimer()
        return
    except ValueError as exc:
        error_message = str(exc)
        with st.chat_message('assistant'):
            st.error(error_message)
        messages.append({'role': 'assistant', 'content': error_message})
        st.session_state.chat_history = trim_chat_history(messages)
        render_disclaimer()
        return

    messages.append({'role': 'assistant', 'content': rendered_answer})
    st.session_state.chat_history = trim_chat_history(messages)
    render_disclaimer()


if __name__ == '__main__':
    main()

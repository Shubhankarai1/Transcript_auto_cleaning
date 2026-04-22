from __future__ import annotations

from typing import Any

import requests
import streamlit as st


API_BASE_URL = "https://iitm-curriculem-intelligence-layer.onrender.com"
REQUEST_TIMEOUT = 60
PAGE_TITLE = "IITM - Curriculum RAG Pipeline"
MODULE_LABELS = {
    "cms": "Contextual Reasoning for Multi-Agent Systems",
    "map": "Multi-Agent Planning & Workflow Design",
    "wdp": "Workflow Design & Optimization",
}


def init_state() -> None:
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}


def build_scope_key(mode: str, module: str | None = None, session: int | None = None) -> str:
    if mode == "global":
        return "global"
    return f"filtered::{module}::{session}"


def get_module_label(module_key: str) -> str:
    return MODULE_LABELS.get(module_key, module_key)


@st.cache_data(show_spinner=False)
def fetch_modules() -> list[str]:
    response = requests.get(f"{API_BASE_URL}/modules", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        modules = data.get("modules", [])
        return [str(item) for item in modules]
    return []


@st.cache_data(show_spinner=False)
def fetch_sessions(module: str) -> list[int]:
    response = requests.get(
        f"{API_BASE_URL}/sessions",
        params={"module": module},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        return [int(item) for item in data]
    if isinstance(data, dict):
        sessions = data.get("sessions", [])
        return [int(item) for item in sessions]
    return []


def send_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    if isinstance(data, dict):
        answer = data.get("answer") or data.get("response") or data.get("message")
        if isinstance(answer, str) and answer.strip():
            sources = data.get("sources", [])
            if not isinstance(sources, list):
                sources = []
            return {
                "answer": answer,
                "sources": sources,
            }

    raise ValueError("The API response did not include an answer field.")


def format_sources_markdown(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""

    lines = ["Sources retrieved:"]
    for source in sources:
        citation = source.get("citation") or "UNKNOWN"
        module = source.get("module") or "unknown"
        session = source.get("session")
        chunk = source.get("chunk")
        lines.append(
            f"- `{citation}` | module `{module}` | session `{session}` | chunk `{chunk}`"
        )
    return "\n".join(lines)


def ensure_history(scope_key: str) -> list[dict[str, str]]:
    histories = st.session_state.chat_histories
    if scope_key not in histories:
        histories[scope_key] = []
    return histories[scope_key]


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
        "<div class='hero-subtitle'>Advanced Engineering Program in AI Agent Workflows & Agentic Systems Development</div>",
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(
            """
            <div class='hero-description'>
                This AI-powered RAG (Retrieval-Augmented Generation) system is built on selected advanced modules, enabling you to interact with highly focused learning content through natural language.
            </div>
            <div class='hero-description'>
                The system currently includes structured knowledge from:
            </div>
            <div class='hero-description'>
                <strong>Module 7: Contextual Reasoning for Multi-Agent Systems</strong><br>
                (Sessions 1, 2, 3, and 4)<br><br>
                <strong>Module 8: Multi-Agent Planning &amp; Workflow Design</strong><br>
                (Sessions 2 and 3)<br><br>
                <strong>Module 9: Workflow Design &amp; Optimization</strong><br>
                (Sessions 1, 2, 3, 4, and 5)
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            With this setup, you can:

            - Explore deep concepts specifically within advanced agentic system design
            - Ask questions across multiple sessions for connected understanding
            - Drill down into individual sessions for precise, context-aware answers
            - Understand how reasoning, planning, and workflows integrate in multi-agent systems
            """
        )
        st.markdown(
            """
            <div class='hero-footer'>
                The system retrieves relevant insights from these sessions and generates grounded, context-rich responses, helping you grasp complex concepts, resolve doubts, and navigate advanced topics more effectively.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        return
        st.markdown(
            """
            <div class='hero-footer'>
                The system retrieves relevant information from lecture data and generates grounded responses—helping you explore concepts, clarify doubts, and navigate the program more effectively.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")


def render_history(messages: list[dict[str, str]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.write("")


def render_sidebar() -> tuple[str, str | None, int | None]:
    with st.sidebar:
        st.header("Chat Scope")
        chat_mode_label = st.radio(
            "Chat Mode",
            options=["All Content", "Select Module & Session"],
            index=0,
        )

        if chat_mode_label == "All Content":
            return "global", None, None

        try:
            modules = fetch_modules()
        except requests.RequestException as exc:
            st.error(f"Failed to load modules: {exc}")
            return "filtered", None, None

        if not modules:
            st.warning("No modules available.")
            return "filtered", None, None

        selected_module = st.selectbox(
            "Module",
            options=modules,
            format_func=get_module_label,
        )

        try:
            sessions = fetch_sessions(selected_module)
        except requests.RequestException as exc:
            st.error(f"Failed to load sessions: {exc}")
            return "filtered", selected_module, None

        if not sessions:
            st.warning("No sessions available for the selected module.")
            return "filtered", selected_module, None

        selected_session = st.selectbox("Session", options=sessions)
        return "filtered", selected_module, int(selected_session)


def build_payload(question: str, mode: str, module: str | None, session: int | None) -> dict[str, Any]:
    if mode == "global":
        return {
            "question": question,
            "mode": "global",
        }

    return {
        "question": question,
        "mode": "filtered",
        "module": module,
        "session": session,
    }


def render_global_welcome() -> None:
    st.markdown(
        """
        <div class="welcome-card">
            <div style="font-size: 1.05rem; font-weight: 600; margin-bottom: 0.35rem;">
                You are chatting with the entire IITM curriculum knowledge base.
            </div>
            <div style="color: #4b5563;">
                Ask anything across all modules and sessions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_scope_summary(mode: str, module: str | None, session: int | None) -> None:
    if mode != "filtered" or not module or session is None:
        return

    st.markdown(
        f"<div class='info-bar'>Module: {get_module_label(module)} | Session: {session}</div>",
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown("---")
    st.markdown(
        """
        <div class='hero-disclaimer'>
            This content is not owned by me and is shared only for educational purposes. All rights belong to IIT Madras, Futurense, and the respective instructors.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    init_state()

    render_styles()
    render_header()

    mode, module, session = render_sidebar()
    render_scope_summary(mode, module, session)

    if mode == "global":
        render_global_welcome()

    if mode == "filtered" and (not module or session is None):
        st.info("Select a module and session from the sidebar to start chatting within that scope.")
        return

    scope_key = build_scope_key(mode, module, session)
    messages = ensure_history(scope_key)

    with st.container():
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        render_history(messages)
        st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask a question")
    if not prompt:
        render_disclaimer()
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.write("")

    payload = build_payload(prompt, mode, module, session)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = send_chat_request(payload)
                answer = response_data["answer"]
                sources_markdown = format_sources_markdown(response_data["sources"])
                rendered_answer = answer
                if sources_markdown:
                    rendered_answer = f"{answer}\n\n{sources_markdown}"
                st.markdown(rendered_answer)
                st.write("")
    except requests.RequestException as exc:
        error_message = f"Request failed: {exc}"
        with st.chat_message("assistant"):
            st.error(error_message)
        messages.append({"role": "assistant", "content": error_message})
        render_disclaimer()
        return
    except ValueError as exc:
        error_message = str(exc)
        with st.chat_message("assistant"):
            st.error(error_message)
        messages.append({"role": "assistant", "content": error_message})
        render_disclaimer()
        return

    messages.append({"role": "assistant", "content": rendered_answer})
    render_disclaimer()


if __name__ == "__main__":
    main()

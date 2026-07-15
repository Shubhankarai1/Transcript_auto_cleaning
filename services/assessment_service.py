from __future__ import annotations

from typing import Any

from supabase_client import get_supabase_admin_client, is_supabase_configured


UNAVAILABLE = object()

QUESTIONS = [
    {
        'id': 'q1',
        'text': 'What is the most repetitive task you perform daily that you wish a computer could handle?',
        'category': 'ai_awareness',
        'options': [
            {'label': 'Writing routine emails/reports', 'value': 'A'},
            {'label': 'Analyzing data or customer feedback', 'value': 'B'},
            {'label': 'Building custom tools or automated systems', 'value': 'C'},
        ],
    },
    {
        'id': 'q2',
        'text': 'How much time per week do you believe could be saved if you had an AI partner?',
        'category': 'ai_awareness',
        'options': [
            {'label': '1-2 hours', 'value': 'A'},
            {'label': '3-5 hours', 'value': 'B'},
            {'label': '5+ hours', 'value': 'C'},
        ],
    },
    {
        'id': 'q3',
        'text': 'Which department function do you think is currently most behind in using AI?',
        'category': 'ai_awareness',
        'options': [
            {'label': 'General Administration', 'value': 'A'},
            {'label': 'Operations/Project Management', 'value': 'B'},
            {'label': 'IT Infrastructure/Development', 'value': 'C'},
        ],
    },
    {
        'id': 'q4',
        'text': 'How do you currently ask an AI to help you?',
        'category': 'interaction',
        'options': [
            {'label': 'Casual, simple queries', 'value': 'A'},
            {'label': 'Structured, persona-based prompts', 'value': 'B'},
            {'label': 'Iterative chains of prompts and fine-tuned instructions', 'value': 'C'},
        ],
    },
    {
        'id': 'q5',
        'text': 'What do you do when an AI gives you an almost perfect answer?',
        'category': 'interaction',
        'options': [
            {'label': 'I re-type the question or move on', 'value': 'A'},
            {'label': 'I provide more context/instructions to refine it', 'value': 'B'},
            {'label': 'I adjust model settings or chain new prompts to fix it', 'value': 'C'},
        ],
    },
    {
        'id': 'q6',
        'text': 'Have you ever tried to chain multiple AI requests to get a complex outcome?',
        'category': 'interaction',
        'options': [
            {'label': 'No', 'value': 'A'},
            {'label': 'Occasionally', 'value': 'B'},
            {'label': 'Frequently/As part of my workflow', 'value': 'C'},
        ],
    },
    {
        'id': 'q7',
        'text': 'How comfortable are you with terms like API, Model, or RAG?',
        'category': 'technical_comfort',
        'options': [
            {'label': '1-2 (Not comfortable)', 'value': 'A'},
            {'label': '3 (Somewhat comfortable)', 'value': 'B'},
            {'label': '4-5 (Very comfortable)', 'value': 'C'},
        ],
    },
    {
        'id': 'q8',
        'text': 'Do you currently use AI features inside apps (e.g., Copilot in Excel, CRM AI)?',
        'category': 'technical_comfort',
        'options': [
            {'label': 'No, I stick to basic chat', 'value': 'A'},
            {'label': 'Yes, occasionally', 'value': 'B'},
            {'label': 'Yes, I actively leverage them for deep tasks', 'value': 'C'},
        ],
    },
    {
        'id': 'q9',
        'text': 'Are you interested in learning how to connect AI to your own data or spreadsheets?',
        'category': 'technical_comfort',
        'options': [
            {'label': 'Perhaps later', 'value': 'A'},
            {'label': 'Yes, definitely', 'value': 'B'},
            {'label': 'Yes, it is a priority', 'value': 'C'},
        ],
    },
    {
        'id': 'q10',
        'text': 'What is your primary concern when sharing company information with an AI?',
        'category': 'safety_verification',
        'options': [
            {'label': 'Not saying anything wrong/biased', 'value': 'A'},
            {'label': 'Data privacy/company policy', 'value': 'B'},
            {'label': 'Scalability and system vulnerabilities', 'value': 'C'},
        ],
    },
    {
        'id': 'q11',
        'text': 'How do you verify if the information provided by an AI is actually accurate?',
        'category': 'safety_verification',
        'options': [
            {'label': 'I trust the first result', 'value': 'A'},
            {'label': 'I do a quick manual check', 'value': 'B'},
            {'label': 'I perform cross-verification and logic testing', 'value': 'C'},
        ],
    },
    {
        'id': 'q12',
        'text': 'Are you familiar with your company internal guidelines for using AI safely?',
        'category': 'safety_verification',
        'options': [
            {'label': 'No', 'value': 'A'},
            {'label': 'Partially', 'value': 'B'},
            {'label': 'Yes, fully familiar', 'value': 'C'},
        ],
    },
    {
        'id': 'q13',
        'text': 'What is your dream outcome from this training?',
        'category': 'ambition',
        'options': [
            {'label': 'Save time on daily tasks', 'value': 'A'},
            {'label': 'Lead AI projects in my department', 'value': 'B'},
            {'label': 'Build/deploy AI solutions for the team', 'value': 'C'},
        ],
    },
    {
        'id': 'q14',
        'text': 'Do you prefer learning by reading theory or by building actual AI projects?',
        'category': 'ambition',
        'options': [
            {'label': 'Theory/Articles', 'value': 'A'},
            {'label': 'Guided walkthroughs', 'value': 'B'},
            {'label': 'Hands-on project building', 'value': 'C'},
        ],
    },
    {
        'id': 'q15',
        'text': 'Are you aiming to be a user of AI tools or an architect of AI solutions?',
        'category': 'ambition',
        'options': [
            {'label': 'User', 'value': 'A'},
            {'label': 'Integrator', 'value': 'B'},
            {'label': 'Architect', 'value': 'C'},
        ],
    },
]

SCORE_MAP = {'A': 1, 'B': 3, 'C': 5}


def _calculate_score(answers: list[dict[str, str]]) -> int:
    return sum(SCORE_MAP.get(a.get('selected_value', ''), 0) for a in answers)


def _get_track(total_score: int) -> str:
    if total_score <= 35:
        return 'foundations'
    if total_score <= 55:
        return 'practitioner'
    return 'builder'


CATEGORY_LABELS = {
    'ai_awareness': 'AI Awareness & Impact',
    'interaction': 'AI Interaction & Prompting',
    'technical_comfort': 'Technical Comfort',
    'safety_verification': 'AI Safety & Verification',
    'ambition': 'Learning Goals & Ambition',
}


def _compute_category_scores(answers: list[dict[str, str]]) -> dict[str, int]:
    cat_scores: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    for a in answers:
        qid = a.get('question_id', '')
        question = next((q for q in QUESTIONS if q['id'] == qid), None)
        if not question:
            continue
        cat = question['category']
        cat_scores[cat] = cat_scores.get(cat, 0) + SCORE_MAP.get(a.get('selected_value', ''), 0)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    normalized = {}
    for cat, total in cat_scores.items():
        count = cat_counts.get(cat, 1)
        normalized[cat] = round(total / count, 1)
    return normalized


def _determine_strengths(category_scores: dict[str, float]) -> list[str]:
    strengths = []
    for cat, score in category_scores.items():
        label = CATEGORY_LABELS.get(cat, cat)
        if score >= 4.0:
            strengths.append(f'Strong {label.lower()}')
        elif score >= 3.0:
            strengths.append(f'Competent {label.lower()}')
    return strengths or ['Building foundational AI awareness']


def _determine_gaps(category_scores: dict[str, float]) -> list[str]:
    gaps = []
    for cat, score in category_scores.items():
        label = CATEGORY_LABELS.get(cat, cat)
        if score < 2.0:
            gaps.append(f'Develop {label.lower()}')
        elif score < 3.0:
            gaps.append(f'Strengthen {label.lower()}')
    return gaps or ['Continue building on current skills']


def compute_result(answers: list[dict[str, str]]) -> dict[str, Any]:
    total_score = _calculate_score(answers)
    track = _get_track(total_score)
    category_scores = _compute_category_scores(answers)
    strengths = _determine_strengths(category_scores)
    gaps = _determine_gaps(category_scores)

    return {
        'total_score': total_score,
        'recommended_track': track,
        'category_scores': [
            {'category': cat, 'label': CATEGORY_LABELS.get(cat, cat), 'score': score}
            for cat, score in category_scores.items()
        ],
        'strengths': strengths,
        'gaps': gaps,
    }


def get_questions() -> list[dict[str, Any]]:
    return QUESTIONS


def _get_client():
    if not is_supabase_configured():
        return None
    try:
        return get_supabase_admin_client()
    except Exception:
        return None


def submit_assessment(user_id: str, answers: list[dict[str, str]]) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return None

    result = compute_result(answers)
    record = {
        'user_id': user_id,
        'raw_answers': answers,
        'scored_result': result,
        'recommended_track': result['recommended_track'],
    }
    try:
        response = client.table('assessment_attempts').insert(record).execute()
        data = response.data or []
        if data:
            return dict(data[0])
        return record
    except Exception:
        return None


def get_latest_assessment(user_id: str) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return None

    try:
        response = (
            client.table('assessment_attempts')
            .select('*')
            .eq('user_id', user_id)
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


def get_assessment_by_id(attempt_id: str) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        return None

    try:
        response = client.table('assessment_attempts').select('*').eq('id', attempt_id).limit(1).execute()
        data = response.data or []
        if not data:
            return None
        return dict(data[0])
    except Exception:
        return None

# API Surface — MVP

## Existing Endpoints (Already in api.py)

| Method | Path | Purpose | Phase |
|--------|------|---------|-------|
| GET | `/` | Health check | Current |
| GET | `/health` | Health + Supabase config status | Current |
| GET | `/v1/profile` | Fetch user profile by user_id | Current |
| POST | `/v1/profile` | Create user profile | Current |
| PUT | `/v1/profile` | Update user profile | Current |
| GET | `/levels` | List content levels (beginner/intermediate/advanced) | Current |
| GET | `/modules` | List modules, filterable by level | Current |
| GET | `/sessions` | List session numbers for a module | Current |
| POST | `/chat` | RAG chat with query rewrite, expansion, HyDE, retrieval, answer | Current |

These endpoints stay as-is and continue working through the api.py refactor (Phase 2).

---

## New Endpoints — MVP

### Auth

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/auth/signup` | Register with email/password (or delegate to Supabase client-side) |
| POST | `/v1/auth/login` | Login (or delegate to Supabase client-side) |
| POST | `/v1/auth/logout` | Invalidate session |

Decision: If using Supabase Auth with Google login, auth endpoints live in the frontend (Supabase JS client) and the backend only verifies the JWT via middleware. No backend auth routes needed.

### Assessment

| Method | Path | Request | Response | Purpose |
|--------|------|---------|----------|---------|
| GET | `/v1/assessment/questions` | — | `{ questions: Question[] }` | Fetch the question bank |
| POST | `/v1/assessment/submit` | `{ user_id, answers: Answer[] }` | `{ attempt_id, result: AssessmentResult }` | Submit answers, get scored result |
| GET | `/v1/assessment/result/{attempt_id}` | — | `{ attempt_id, result: AssessmentResult }` | Fetch a past result |

```
Question {
  id: string
  text: string
  category: string        // "ai_fundamentals" | "gen_ai" | "prompt_engineering" | ...
  options: { label: string, value: string }[]
}

Answer {
  question_id: string
  selected_value: string
}

AssessmentResult {
  recommended_track: "foundations" | "practitioner" | "builder"
  strengths: string[]
  gaps: string[]
  scores: { category: string, score: number }[]
}
```

### Roadmap

| Method | Path | Request | Response | Purpose |
|--------|------|---------|----------|---------|
| POST | `/v1/roadmap/generate` | `{ user_id }` | `{ plan_id, roadmap: Roadmap }` | Generate a new roadmap from profile + assessment |
| GET | `/v1/roadmap/active` | `?user_id=` | `{ plan_id, roadmap: Roadmap }` | Get the active roadmap for a user |
| GET | `/v1/roadmap/{plan_id}` | — | `{ plan_id, roadmap: Roadmap }` | Get a specific roadmap version |

```
Roadmap {
  track: "foundations" | "practitioner" | "builder"
  modules: {
    module_name: string
    sessions: { session_number: int, title?: string }[]
    rationale: string
  }[]
  weekly_sequence?: { week: int, focus: string }[]
  estimated_duration: string
}
```

### Dashboard

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| GET | `/v1/dashboard/{user_id}` | `DashboardData` | Aggregate: profile + latest assessment + active roadmap |

```
DashboardData {
  profile: Profile
  latest_assessment?: AssessmentResult
  active_roadmap?: Roadmap
  mentor_intro: string     // optional LLM-generated greeting based on context
}
```

---

## Refactored Route Structure (After Phase 2)

```
api/
  routes/
    __init__.py
    health.py       → GET /, /health
    catalog.py      → GET /levels, /modules, /sessions
    chat.py         → POST /chat
    profiles.py     → GET/POST/PUT /v1/profile
    assessment.py   → GET /v1/assessment/questions, POST /v1/assessment/submit
    roadmap.py      → POST /v1/roadmap/generate, GET /v1/roadmap/active, /v1/roadmap/{id}
    dashboard.py    → GET /v1/dashboard/{user_id}
```

Auth middleware and shared dependencies live in `api/__init__.py` or `api/deps.py`.

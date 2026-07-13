# MVP Feature Checklist

Based on `PRD V2.md` and `phases_2.md`. Each phase is marked **MVP** or **Defer** with the specific deliverables that are in or out.

## Phase 0 — Content System Restructure
**Decision: ✅ MVP (Complete)**

Done. No further work needed before shipping.

---

## Phase 1 — Foundation & Architecture Alignment
**Decision: ✅ MVP (In Progress)**

Deliverables:
- [x] Architecture boundaries document (`Dev Docs/architecture.md`)
- [x] MVP feature checklist (`Dev Docs/mvp-checklist.md`)
- [x] Entity relationship draft (`Dev Docs/database_schema.md`)
- [x] Deployment topology: **single FastAPI + Streamlit**
- [x] Canonical env vars documented (`Dev Docs/environment-variables.md`)
- [x] API surface draft (`Dev Docs/api-surface.md`)

All Phase 1 deliverables are complete.

---

## Phase 2 — Backend Refactor Into Platform Services
**Decision: ✅ MVP**

Split `api.py` into modules, but only the services needed for MVP:
- `/api/routes/` — chat, catalog, health, profiles
- `/api/rag/` — pipeline, search, sources (extracted from `api.py` + `query_rag.py`)
- `/api/llm/` — OpenAI client helpers

Out of scope for MVP: assessment service, roadmap service. Those routes will be added in Phase 5/7 but the module structure should accommodate them.

---

## Phase 3 — Authentication & User Data Layer
**Decision: ✅ MVP**

In scope:
- Supabase Auth (Google login or email)
- Auth middleware or helper in FastAPI
- Frontend login / logout / session
- `users` table

---

## Phase 4 — Learner Profile Onboarding
**Decision: ✅ MVP**

In scope:
- Multi-step onboarding form (role, industry, experience, goals, availability)
- Save/resume behavior
- Profile edit screen
- Backend CRUD for `learner_profiles`
- Require profile before accessing dashboard

---

## Phase 5 — AI Readiness Assessment Engine
**Decision: ✅ MVP**

In scope:
- Assessment question bank (from `skill mapping questions.md` or similar)
- Deterministic scoring (rule-based, no LLM for classification)
- Result: overall readiness level + strengths + gaps + recommended track
- Store raw answers + scored output
- Backend: submit, score, fetch result

---

## Phase 6 — Learning Track Model & Content Mapping
**Decision: ✅ MVP**

In scope:
- Three track definitions: AI Foundations, AI Practitioner, AI Builder
- Track-to-content mapping: which modules/sessions belong to each track
- Role-to-track mapping (HR → Foundations, PM → Practitioner, etc.)
- Store mappings in Supabase tables (not hardcoded)

This depends on Phase 0 being complete (it is) and is required by Phase 7.

---

## Phase 7 — Personalized Roadmap Generator
**Decision: ✅ MVP**

In scope:
- Deterministic rule-based roadmap generation (profile + assessment + track mapping)
- Roadmap JSON: track, modules, sessions, sequence, milestones, duration, rationale
- Save and fetch roadmap
- LLM used only for natural-language explanation, not selection logic

---

## Phase 8 — Dashboard & Guided Learning Experience
**Decision: ✅ MVP**

In scope:
- Dashboard as main screen (not chat-first)
- Sections: learner summary, current track, assessment result, roadmap progress, AI Mentor entry
- Roadmap page, assessment result page, profile page, mentor workspace
- Navigation between pages

---

## Phase 9 — AI Mentor Integration With Learner Context
**Decision: ✅ MVP**

In scope:
- Pass learner context into mentor (role, track, roadmap stage, gaps)
- Level-aware retrieval preference (beginner/intermediate/advanced chunks)
- Mentor can explain concepts, summarize next step, recommend what to study
- Existing RAG grounding preserved

---

## Phase 10 — Admin Operations & Content Lifecycle
**Decision: ❌ Defer until after MVP**

Deferred items:
- Admin UI for adding modules / tracks
- Versioning for assessments and roadmaps
- Auditability for user progress

---

## Phase 11 — QA, Evaluation & Release Readiness
**Decision: ✅ MVP (Reduced)**

In scope for MVP:
- Manual acceptance walkthrough of the full learner flow
- Regression check on existing RAG endpoints
- Seed test users for each track

Deferred:
- Automated test coverage
- Formal QA phase

---

## MVP Acceptance Flow

```
1.  User signs in (Google / Supabase Auth)
2.  User completes profile onboarding
3.  User lands on dashboard
4.  User completes AI readiness assessment
5.  System recommends a track (Foundations / Practitioner / Builder)
6.  System generates a learning roadmap
7.  User opens AI Mentor with personalised context
8.  User receives grounded guidance linked to their roadmap
```

## Summary Table

| Phase | Decision | Status |
|-------|----------|--------|
| 0 — Content Restructure | ✅ MVP | Complete |
| 1 — Architecture Alignment | ✅ MVP | Complete |
| 2 — Backend Refactor | ✅ MVP | Not started |
| 3 — Authentication | ✅ MVP | Not started |
| 4 — Profile Onboarding | ✅ MVP | Not started |
| 5 — Assessment Engine | ✅ MVP | Not started |
| 6 — Track & Content Mapping | ✅ MVP | Not started |
| 7 — Roadmap Generator | ✅ MVP | Not started |
| 8 — Dashboard UX | ✅ MVP | Not started |
| 9 — Mentor Personalisation | ✅ MVP | Not started |
| 10 — Admin / Content Lifecycle | ❌ Defer | — |
| 11 — QA & Release | ✅ MVP (reduced) | Not started |

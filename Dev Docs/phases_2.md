# Project Phases v2: Execution Plan For AI Learning Mentor

This file translates `PRD V2.md` into an execution roadmap for extending the current IITM transcript RAG project into a full AI Learning Mentor platform.

The goal is not to rebuild the knowledge layer. The existing transcript ingestion, Pinecone retrieval, and AI Mentor chat backend remain the shared AI knowledge service. The work below focuses on the application layer that sits around it.

## Current Baseline

Already available in this repository:

- Transcript ingestion and cleaning pipeline
- Hierarchy-aware content ingestion for level, category, module, and session
- RAG-ready chunk generation with level/category/module/session metadata
- Pinecone upload workflow with structure-aware vector metadata
- FastAPI backend for retrieval and chat
- Streamlit frontend for AI Mentor chat
- Basic level/module scoped learning support
- Basic profile persistence endpoints backed by Supabase helpers

What is missing for PRD v2:

- Authentication
- Persistent user data model
- Learner onboarding and profile management
- AI readiness assessment flow
- Personalized roadmap generation
- Dashboard and guided journey UX
- Production-grade platform architecture around the existing AI layer

## Execution Principles

- Keep the current RAG service as an independent internal capability.
- Treat content structure as the first dependency, because roadmap logic and mentor behavior depend on what content exists at each level.
- Add platform features in thin vertical slices instead of large rewrites.
- Introduce persistence early so onboarding, assessment, and roadmap features share the same user record.
- Avoid coupling roadmap logic directly to transcript retrieval logic.
- Ship an MVP that completes the end-to-end learner journey before optimizing analytics or adaptive learning.

## Phase 0: Content System Restructure

### Objective

Restructure the current content layer so the platform can support beginner, intermediate, and advanced learning content instead of only the current advanced transcript corpus.

### Why This Must Come First

- The roadmap engine cannot recommend beginner or intermediate learning paths if the content base only contains advanced material.
- The mentor cannot reliably answer by learner level unless chunks carry level-aware metadata.
- The frontend and backend should not be designed around the current advanced-only content assumption.

### Deliverables

- Define the future content hierarchy under `input/`
- Separate content by learner level using the current agreed structure:
  - `input/level_1_foundations/`
  - `input/level_2_intermediate/`
  - `input/level_3_advanced/`
- For `level_1_foundations`, support:
  - `common_modules/`
  - `subject_matter_expertise/`
- For `level_2_intermediate`, support:
  - `common_modules/`
  - `role_specific/`
- For `level_3_advanced`, keep the existing open module structure:
  - `cms/`
  - `map/`
  - `wdp/`
- Move existing advanced transcripts into `level_3_advanced/`
- Restore level 1 and level 2 to a structured module layout:
  - category folder
  - module folder
  - `session_<n>.txt`
- Update the ingestion assumptions so content is no longer flat transcript-only

### Current Status

Completed:

- The `input/` folder has been restructured into:
  - `level_1_foundations`
  - `level_2_intermediate`
  - `level_3_advanced`
- `level_1_foundations` now uses:
  - `common_modules/<module_name>/session_<n>.txt`
  - `subject_matter_expertise/<module_name>/session_<n>.txt`
- `level_2_intermediate` now uses:
  - `common_modules/<module_name>/session_<n>.txt`
  - `role_specific/<module_name>/session_<n>.txt`
- `level_3_advanced` now uses:
  - `cms/session_<n>.txt`
  - `map/session_<n>.txt`
  - `wdp/session_<n>.txt`
- Existing advanced transcript content has already been placed under `level_3_advanced`
- A structure note has been added at `input/README_STRUCTURE.md`
- The ingestion pipeline now reads the new hierarchy using recursive discovery under `input/`
- File path parsing now extracts:
  - level
  - category
  - module
  - session number
  - hierarchy-aware identifiers
- Module outputs now preserve `Level` and `Category` headers
- RAG chunk generation now preserves:
  - level
  - category
  - module
  - topic
  - session
  - chunk id
- Pinecone upload now includes:
  - level
  - category
  - module
  - session
  - chunk
  - topic-derived metadata
- `level_3_advanced` remains compatible without requiring a category layer
- `module_path` and session-specific `content_id` now flow through regenerated `rag_chunks/` and refreshed Pinecone upload metadata

Not completed yet:

- A fresh validation run should still be done whenever content is changed, so regenerated `rag_chunks/` and Pinecone data stay fully aligned with the current structure

### Required Pipeline Upgrades

- Done:
  - Content loading now discovers files from nested level-based folders
  - Structure-aware fields are extracted from the file path
  - Chunk generation now preserves level/category/module/topic/session metadata
  - Pinecone upload now includes the new hierarchy metadata
  - `level_3_advanced` is supported without a category layer
- Remaining:
  - Rebuild or refresh chunk outputs whenever transcript content changes
  - Verify a small end-to-end ingest after each structural change
  - Keep validating regenerated chunk outputs and Pinecone metadata whenever transcript content is updated

### Immediate Implementation Rule

- Hierarchy-aware labeling is now implemented.
- Future full ingests should use the updated pipeline so regenerated chunks and Pinecone vectors stay consistent with the current content tree.

### Exit Criteria

- The repository supports beginner, intermediate, and advanced source content
- Chunks can be generated from the new structure
- The vector store schema can distinguish content by:
  - level
  - category
  - module
- A small test ingest validates that hierarchy labels flow correctly into Pinecone metadata

Status:

- Core implementation is complete.
- Regenerated chunk outputs and refreshed Pinecone metadata now reflect the current hierarchy-aware structure.

## Phase 1: Foundation And Architecture Alignment

### Objective

Stabilize the current codebase and define the target architecture before adding platform features.

### Deliverables

- Finalize product scope for MVP versus post-MVP
- Confirm system ownership boundaries:
  - AI knowledge layer
  - application backend
  - frontend experience
  - persistence/auth layer
- Define canonical environment variables and service configuration
- Decide deployment topology:
  - single FastAPI service plus Streamlit
  - or Streamlit as frontend and FastAPI as separate app service
- Create a basic domain model for:
  - users
  - learner profiles
  - assessments
  - roadmap recommendations
  - chat sessions

### Recommended Output

- Architecture note
- Entity relationship draft
- API surface draft
- MVP feature checklist

### Current Status

- [x] Architecture boundaries documented (`Dev Docs/architecture.md`)
- [x] MVP feature checklist documented (`Dev Docs/mvp-checklist.md`)
- [x] Entity relationship draft documented (`Dev Docs/database_schema.md`)
- [x] API surface draft documented (`Dev Docs/api-surface.md`)
- [x] Canonical environment variables documented (`Dev Docs/environment-variables.md`)
- [x] Deployment topology: single FastAPI + Streamlit
- [x] Configuration loading is centralized through `config.py`
- [x] Local and deployment environment separation is in place through `.env` and `.env.local`
- All Phase 1 deliverables are complete.
- Phase 2 (backend refactor) is the next step.

### Exit Criteria

- Team agrees on MVP boundaries
- Supabase is confirmed as the system of record for user and app data
- Existing RAG APIs are identified as reusable internal services

## Phase 2: Backend Refactor Into Platform Services

### Objective

Refactor the current backend from a single-purpose chat API into a platform backend with clear service boundaries.

### Deliverables

- Separate modules for:
  - auth integration
  - profile service
  - assessment service
  - roadmap service
  - mentor/chat service
- Introduce configuration management for:
  - model provider
  - Pinecone index
  - app URLs
  - auth settings
- Keep the current `/chat`, `/modules`, and `/sessions` routes working
- Add versioned platform routes such as:
  - `/v1/profile`
  - `/v1/assessment`
  - `/v1/roadmap`
  - `/v1/dashboard`

### Notes

- This phase should mainly improve structure, not user-facing functionality.
- Do not merge assessment logic or roadmap generation directly into `api.py`.

### Current Status

- The backend still centers on `api.py`, so this phase is not complete
- Existing reusable platform-facing pieces already present:
  - `/chat`
  - `/levels`
  - `/modules`
  - `/sessions`
  - `/v1/profile`
- Configuration and service helpers already exist, but the backend is not yet refactored into clear modules for auth, assessment, roadmap, and mentor services

### Exit Criteria

- The backend is modular enough to support user-centric features without turning `api.py` into a monolith.

## Phase 3: Authentication And User Data Layer

### Objective

Enable secure sign-in and persistent user records.

### Deliverables

- Choose auth approach:
  - Supabase Auth only
  - or Google login through Supabase
- Create Supabase schema for:
  - users
  - learner_profiles
  - assessment_attempts
  - roadmap_plans
  - mentor_conversations
- Add backend middleware or helper layer for authenticated requests
- Add frontend login and session handling

### Suggested Initial Data Model

- `users`
  - auth provider id
  - email
  - created_at
- `learner_profiles`
  - user_id
  - current_role
  - industry
  - years_experience
  - career_aspirations
  - ai_learning_goals
  - weekly_learning_availability
- `assessment_attempts`
  - user_id
  - raw_answers
  - scored_result
  - recommended_track
  - created_at
- `roadmap_plans`
  - user_id
  - track
  - roadmap_json
  - version
  - created_at

### Exit Criteria

- A user can sign in, sign out, and persist a profile record across sessions.

### Current Status

- Partial progress exists:
  - Supabase client helpers are implemented
  - profile create, read, and update endpoints exist at `/v1/profile`
  - profile fields and persistence scaffolding are present
- Authentication itself is not implemented yet
- Frontend login and authenticated session handling are not implemented yet

## Phase 4: Learner Profile Onboarding

### Objective

Build the first-run onboarding flow that captures the learner context required for personalization.

### Deliverables

- Multi-step onboarding form
- Validation and save/resume behavior
- Profile edit screen
- Backend endpoints to create, update, and fetch profile state

### Product Rules

- Onboarding must be short enough to complete quickly
- Every field should have downstream use in assessment or roadmap generation
- Users should not access the full dashboard until minimum profile data is captured

### Exit Criteria

- A first-time user can complete onboarding and land on a personalized dashboard shell.

## Phase 4.5: UI/UX Redesign

### Objective

Redesign the application to feel like a modern SaaS product while keeping it fully compatible with Streamlit. The full spec is in `Dev Docs/design.md`.

### Deliverables

- Apply theme (background `#FAFAFA`, primary `#2563EB`, accent `#14B8A6`, 12px border-radius, soft shadows, 1000px max-width centered)
- Add a landing page before login (hero with Get Started / Login buttons, 3 feature cards, How It Works 4-step section, footer)
- Redesign login page as a centered card (max 450px)
- Redesign onboarding as centered cards (max 700px) with a progress bar, dropdowns/radios instead of text areas
- Clean up AI Mentor page: remove long intro paragraphs, show a single welcome message, tidy sidebar
- Consistent typography and spacing throughout

### Implementation Notes

- Entirely CSS + Streamlit native components. No backend changes, no new libraries.
- Estimated effort: 2–3 hours.

### Exit Criteria

- The app looks clean, centered, and presentable — like a modern SaaS product.
- No backend changes required.

## Phase 5: AI Readiness Assessment Engine

### Objective

Turn the assessment concept into a structured scoring system that maps learners to tracks and gap areas.

### Deliverables

- Convert the assessment questionnaire into a normalized question bank
- Define scoring logic for:
  - AI Foundations
  - AI Practitioner
  - AI Builder
- Support assessment submission, scoring, persistence, and result retrieval
- Show results in a usable format:
  - overall readiness level
  - strengths
  - gaps
  - recommended track

### Technical Design Guidance

- Keep scoring deterministic first
- Use LLM assistance only for explanation or summary, not for the base classification logic
- Store both raw answers and derived scoring outputs

### Exit Criteria

- The system can consistently assign a recommended learning track from a completed assessment.

## Phase 6: Learning Track Model And Content Mapping

### Objective

Map the existing IITM knowledge base into structured learning paths that can power recommendations.

### Deliverables

- Formal track definitions for:
  - AI Foundations
  - AI Practitioner
  - AI Builder
- Define how learner tracks map onto the multi-level content system:
  - beginner content
  - intermediate content
  - advanced content
- Create mapping tables from:
  - user role categories
  - assessment gaps
  - target track
  - recommended modules and sessions
- Normalize learning content metadata so roadmap generation is not based on free text alone

### Recommended Content Structure

- `learning_tracks`
- `track_modules`
- `track_sessions`
- `skill_gaps`
- `role_to_track_mappings`

### Notes

- This phase is where documents like `learning Tracks.md` and `skill mapping questions.md` should be operationalized into structured product data.
- This phase depends on Phase 0 being complete, because track design must map to actual content that exists in the knowledge base.
- If needed, store a curated track catalog in Supabase rather than hardcoding it in Streamlit.

### Exit Criteria

- The platform can programmatically recommend modules and sessions for each track.

## Phase 7: Personalized Roadmap Generator

### Objective

Generate a learner-specific roadmap using profile data, assessment output, and the structured content map.

### Deliverables

- Rule-based roadmap generation for MVP
- Roadmap JSON structure including:
  - selected track
  - recommended modules
  - recommended sessions
  - weekly sequence
  - milestones
  - estimated duration
  - rationale
- Ability to save and fetch the active roadmap

### Recommended MVP Strategy

- Use deterministic recommendation rules first
- Use LLM support only for:
  - natural language explanation
  - personalized mentoring summary
- Keep the selection logic inspectable and testable

### Exit Criteria

- Every onboarded user with a completed assessment can receive a persisted roadmap.

## Phase 8: Dashboard And Guided Learning Experience

### Objective

Replace the current single-purpose chat screen with a dashboard-centered learning product.

### Deliverables

- Dashboard home with:
  - learner summary
  - current track
  - assessment result
  - roadmap progress snapshot
  - AI Mentor entry point
- Roadmap page
- Assessment result page
- Profile page
- Mentor workspace page

### UX Rules

- The dashboard should orient the learner before exposing chat
- AI Mentor should feel like one feature within the platform, not the whole platform
- The roadmap should link directly into recommended modules or mentor prompts

### Exit Criteria

- A user can navigate from dashboard to profile, assessment result, roadmap, and AI Mentor without leaving the product flow.

## Phase 9: AI Mentor Integration With Learner Context

### Objective

Upgrade the existing mentor chat so it is aware of the user journey, not just raw content retrieval.

### Deliverables

- Pass learner context into mentor interactions:
  - role
  - target track
  - current roadmap stage
  - assessment gaps
- Add learner-level awareness to retrieval and response shaping:
  - prefer beginner chunks for beginner learners
  - prefer intermediate chunks for intermediate learners
  - prefer advanced chunks for advanced learners
- Add mentor actions such as:
  - explain this concept for my level
  - summarize my next learning step
  - recommend what to study next
- Preserve the existing grounded RAG behavior for educational accuracy

### Guardrails

- Learner context should shape tone and guidance, not override source grounding
- Mentor recommendations should reference the roadmap when possible
- Unsupported answers should still fail safely

### Exit Criteria

- Mentor responses feel personalized while remaining grounded in the IITM knowledge layer.

## Phase 10: Persistence, Admin Operations, And Content Lifecycle (Optional)

### Objective

Make the platform easier to maintain as content, users, and recommendations evolve.

### When To Include This

Include this only if you want to show a more complete system design beyond the core course-project workflow.

### Deliverables

- Admin-friendly workflow for:
  - adding new modules
  - adding new beginner/intermediate/advanced content
  - updating track mappings
  - refreshing roadmaps when logic changes
- Versioning for:
  - assessments
  - roadmap templates
  - recommendation logic
- Basic auditability for user progress events and roadmap generation

### Exit Criteria

- The product can be updated without fragile manual edits across code and documents.

## Phase 11: QA, Evaluation, And Release Readiness (Optional)

### Objective

Validate the platform end to end before broader rollout.

### When To Include This

For a course project, this can be reduced to simple final validation and demo preparation instead of a full QA phase.

### Deliverables

- Test coverage for:
  - content ingestion by learner level
  - onboarding flow
  - assessment scoring
  - roadmap generation
  - mentor chat integration
- Seed test users for each track
- Regression checks for the current RAG endpoints
- Manual acceptance checklist aligned to the PRD user journey

### MVP Acceptance Flow

1. User signs in
2. User completes profile onboarding
3. User completes AI readiness assessment
4. System recommends a track
5. System generates a roadmap
6. User opens AI Mentor with personalized context
7. User receives grounded guidance linked to the roadmap

### Exit Criteria

- The complete learner flow works reliably in staging and production.

## Recommended Build Order

1. Phase 0: content system restructure
2. Phase 1: foundation and architecture alignment
3. Phase 2: backend refactor into services
4. Phase 3: authentication and persistence
5. Phase 4: learner profile onboarding
6. Phase 5: readiness assessment engine
7. Phase 6: content and track mapping
8. Phase 7: roadmap generator
9. Phase 8: dashboard UX
10. Phase 9: mentor personalization
11. Phase 10: optional admin and lifecycle support
12. Phase 11: optional QA and release readiness

## MVP Cut Recommendation

If execution capacity is limited, define MVP as:

- Multi-level content restructure and re-ingestion
- Authentication
- Learner profile onboarding
- One production assessment flow
- Deterministic track recommendation
- Deterministic roadmap generation
- Dashboard shell
- AI Mentor integration with learner context

Defer these until after MVP:

- adaptive roadmap updates
- ongoing progress analytics
- periodic reassessment
- notifications
- gamification
- advanced admin tooling
- deeper admin lifecycle features
- formal release-readiness work

## Main Risks

- Designing learner pathways before the content base actually supports those levels
- Over-coupling product logic to the current Streamlit prototype
- Letting LLM prompts replace deterministic product logic where rules are more appropriate
- Building personalization before establishing clean content mappings
- Expanding frontend scope before user/auth/persistence are stable
- Breaking the existing RAG assistant while layering in platform features

## Immediate Next Actions

1. Update the ingestion and chunking pipeline so `level`, `category`, `module`, and `session` metadata flow into `rag_chunks/` and Pinecone.
2. Add the `level_3_advanced` fallback rule so `cms`, `map`, and `wdp` are treated as `advanced` even without extra labels.
3. Backfill the existing advanced Pinecone records with `level=advanced` metadata instead of deleting the database.
4. Run a small validation ingest using one sample module from each level.
5. After validation, run the full ingestion for the complete restructured content base.
6. Convert `PRD V2.md` into a scoped MVP backlog with clear in-scope and out-of-scope decisions.
7. Choose the auth and persistence implementation in Supabase.
8. Define the database schema and platform API contracts.
9. Refactor the current backend so mentor chat becomes one service inside a larger application.
10. Turn the assessment and learning track documents into structured data assets.




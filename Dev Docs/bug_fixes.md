# Bug Fixes Log

## Deployment & Infrastructure

### API_BASE_URL pointing to production instead of local
When running locally, the Streamlit frontend was calling the production Render URL (`https://iitm-curriculem-intelligence-layer.onrender.com`) because no `API_BASE_URL` was set in `.env.local`. Added `API_BASE_URL=http://localhost:8000` to `.env.local` so local development hits the local backend.

### FastAPI app not exposed in main.py
`main.py` had `from api import app` but uvicorn was configured to run `main:app` without confirming the FastAPI app was properly initialized. Fixed by ensuring api.py initializes `app = FastAPI()` and main.py re-exports it.

### Deployment error — re-ranker too heavy for free Render plan
The re-ranker used `sentence-transformers` which exceeded the memory limits of Render's free plan. Removed the re-ranker dependency and rely on Pinecone vector search + metadata post-filtering instead.

### Port issue sorted
Resolved port binding conflicts during deployment.

### Base URL error fixed (Streamlit)
Streamlit was hitting the wrong base URL for API calls. Fixed the URL configuration in `app.py`.

## Retrieval & Chat

### "Not in module" false positive
The chat was incorrectly returning "Not in module" when relevant content existed. Fixed the retrieval fallback logic so it widens gradually (session → module → global) instead of giving up early.

### Filtered-to-global fallback not working
Filtered retrieval (by level/module/session) was not falling back to global when results were weak. Implemented a cascading fallback: level+module+session → level+module → level → module → global.

### Re-ranking top_k too high
Top K was set to 25, causing excessive noise in results. Reduced to K=10 for cleaner results.

### Metadata display cluttering the answer
Sources and debug info (HyDE query, retrieval queries, matched_query, raw scores, JSON payloads) were displayed inline with the answer. Moved them into a collapsible section closed by default.

### Metadata UI not rendering properly
The collapsible source display had formatting issues. Fixed the markdown/styling so sources render correctly inside the expander.

### Citations not appearing in answers
Citations like `[MAP-S3-C42]` were not being generated or displayed. Added citation formatting in `build_source_entry()` and inline citation guidance in the LLM prompt.

## Frontend (Streamlit)

### Profile save failing silently with no error detail
`save_profile_to_backend()` caught all exceptions silently. Added detailed error messages showing the HTTP status code, response body, and connection errors.

### Onboarding steps making redundant HTTP calls
Every "Next" button click triggered `PUT /v1/profile` + `st.rerun()` → `GET /v1/auth/me` + `GET /v1/profile`. Removed the redundant verification and profile fetch on rerun — data stays in `st.session_state` between steps.

### Onboarding saving to backend on every step
Partial form data was being saved to Supabase after each of the 3 onboarding steps, causing 3 sequential network calls. Changed to browser-local storage (`st.session_state`) during the form — only the final "Complete Setup" saves to the backend.

### st.set_page_config called multiple times
Both `render_main_app()` and `render_auth_page()` called `st.set_page_config()`, causing Streamlit errors when the token expired mid-session. Moved to a single top-level call before the auth check.

### Token expiry not detected
Users with expired tokens would see the app as "logged in" but all API calls would fail. Added `_verify_token_with_backend()` call at startup that calls `GET /v1/auth/me` — if it fails, auto-logout and redirect to sign-in.

## Backend

### api.py monolithic (687 lines)
All routes, retrieval logic, profile CRUD, and content catalog were in one file. Refactored into `services/` (mentor_service, profile_service, catalog_service) and `routers/` (health, catalog, chat, v1_profiles, v1_auth). api.py reduced to 7 lines.

### ProfileUpsertRequest requiring user_id when JWT is present
The API required `user_id` in the request body even when the JWT token already identifies the user. Made `user_id` optional — when a valid JWT is provided, the token's identity takes precedence.

### /v1/auth/me returning minimal info
The endpoint only returned `id` and `email`. Enriched response with `created_at`, `last_sign_in_at`, `phone`, `app_metadata`, `user_metadata`.

## Ingestion Pipeline

### File structure not matching hierarchy
Transcript files were in a flat structure instead of the `level_<n>/<category>/<module>/session_<n>.txt` hierarchy. Restructured input folder into `level_1_foundations`, `level_2_intermediate`, `level_3_advanced` with proper category/module subdirectories.

### Metadata not flowing into Pinecone
Level, category, module, and session metadata were not being preserved in the chunk generation → Pinecone upload pipeline. Updated `rag_chunks/` generation and Pinecone metadata to include all hierarchy fields.

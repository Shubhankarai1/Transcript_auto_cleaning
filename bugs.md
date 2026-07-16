# Bug Log

## KeyError in onboarding Step 3 — `weekly_learning_availability` missing from profile

- **File:** `app.py:361`
- **Symptom:** `KeyError: 'weekly_learning_availability'` when clicking Submit on onboarding Step 3 of 3.
- **Root cause:** Ternary expression `HOURS_OPTIONS.index(profile['weekly_learning_availability']) if profile.get(…, '') in HOURS_OPTIONS else 0` — the condition uses `.get()` which returns `''` for a missing key, and `''` is in `HOURS_OPTIONS`, so the True branch is taken and the direct bracket access `profile['weekly_learning_availability']` raises a `KeyError`.
- **Fix:** Changed `profile['weekly_learning_availability']` to `profile.get('weekly_learning_availability', '')` on line 361.

## Page navigation requires double-click — sidebar bounces back to Dashboard

- **File:** `app.py`
- **Symptom:** Single-clicking any page in the sidebar nav jumps back to the original page; needs a second click to work.
- **Root cause:** (1) `st.query_params['p'] = ...` in sidebar (line 629) and 5 other locations triggered browser URL updates, which caused Streamlit to fire unsolicited second reruns. On that second rerun, the sidebar radio widget hadn't received the updated state yet and fell back to its `index=` parameter (the original page). (2) `_recover_session()` overrode `st.session_state.page` on every interactive rerun using the stale `?p` value.
- **Fix:** (1) Removed ALL `st.query_params['p'] = ...` assignments (6 total) — prevents all URL-triggered double reruns. (2) Changed `_recover_session()` to only restore page on initial load (`if 'page' not in st.session_state`) instead of every rerun — prevents stale `?p` from overriding the current page. (3) Added explicit `key='nav_radio'` to the sidebar navigation radio widget for stable widget-state tracking.

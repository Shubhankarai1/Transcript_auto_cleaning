# Bug Log

## KeyError in onboarding Step 3 — `weekly_learning_availability` missing from profile

- **File:** `app.py:361`
- **Symptom:** `KeyError: 'weekly_learning_availability'` when clicking Submit on onboarding Step 3 of 3.
- **Root cause:** Ternary expression `HOURS_OPTIONS.index(profile['weekly_learning_availability']) if profile.get(…, '') in HOURS_OPTIONS else 0` — the condition uses `.get()` which returns `''` for a missing key, and `''` is in `HOURS_OPTIONS`, so the True branch is taken and the direct bracket access `profile['weekly_learning_availability']` raises a `KeyError`.
- **Fix:** Changed `profile['weekly_learning_availability']` to `profile.get('weekly_learning_availability', '')` on line 361.

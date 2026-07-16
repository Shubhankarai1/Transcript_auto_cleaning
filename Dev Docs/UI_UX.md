# UI/UX Beautification Plan

Current audit score: **6.5 / 10**

---

## Risk Note

All changes here are **presentational only** — no business logic, API calls, or data flow is touched. Every change is revertible with `git checkout`. Worst case is a visual regression, never a crash.

---

## Phase 1: Quick Wins (15 min, zero breakage risk)

### 1.1 Fix CSS hex alpha bug

**What:** `{color}15` on track badge backgrounds (e.g. `#10b98115`) is invalid CSS and renders literally.

**Where:** Lines 713, 1086, 1502 in `app.py`.

**Fix:** Replace inline style `background: {track_color}15` with:
```python
f"background: {track_color}; opacity: 0.1;"
```

**Files touched:** `app.py` only.

---

### 1.2 Add loading spinner on assessment submit

**What:** User clicks Submit on the 15-question assessment and waits with zero feedback.

**Where:** `assessment_page()` around line 1204.

**Fix:** Wrap the API call in `with st.spinner('Scoring your assessment...')`.

**Files touched:** `app.py` only.

---

### 1.3 Remove dead `sidebar()` function

**What:** `sidebar()` at line 501 is defined but never called (replaced by `render_nav_sidebar()`).

**Fix:** Delete the unused function.

**Files touched:** `app.py` only.

---

### 1.4 Fix duplicate `.block-container` rule

**What:** `padding-top` is set twice — first `0.5rem`, then overridden to `1.5rem`.

**Fix:** Keep only the second rule, remove the first.

**Files touched:** `app.py` only.

---

## Phase 2: Theme & Structure (30 min)

### 2.1 Add `.streamlit/config.toml`

**What:** No theme config exists. Native Streamlit components (buttons, selects, progress bars) use defaults that may not match the custom CSS.

**Fix:** Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#fafafa"
secondaryBackgroundColor = "#ffffff"
textColor = "#111827"
font = "sans serif"
```

**Files touched:** New file `.streamlit/config.toml`.

**Risk:** 🟢 Minimal. Streamlit ignores missing config. Theme only affects native widgets — custom HTML cards are untouched.

---

### 2.2 Add `border-box` global reset

**What:** Inline padding on some cards overflows their containers on narrow screens.

**Fix:** Add to `inject_css()`:
```css
* { box-sizing: border-box; }
```

**Files touched:** `app.py` only.

---

### 2.3 Consolidate repeated inline styles into CSS classes

**What:** ~50 inline `style=` blocks across 7 pages. Hard to maintain and audit.

**Fix:** Add reusable CSS classes to `inject_css()` for the most common patterns:

```css
.score-number { font-size: 3rem; font-weight: 800; }
.score-label { font-size: 1rem; color: #6b7280; }
.track-pill { display: inline-block; padding: 0.25rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem; color: white; }
.stat-value { font-size: 2rem; font-weight: 800; }
.stat-label { font-size: 0.85rem; color: #9ca3af; }
.module-mini-card { padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.category-bar { background: #e5e7eb; border-radius: 8px; height: 10px; }
.category-bar-fill { border-radius: 8px; height: 10px; }
.page-heading { margin-bottom: 0.25rem; }
.page-subtitle { color: #6b7280; margin-bottom: 1.5rem; }
```

Then replace the corresponding inline styles in each page function.

**Files touched:** `app.py` — `inject_css()` + each page function.

**Risk:** 🟡 Low-Medium. Each replacement must be verified — a missed class means a visual regression on that one element. Use search-and-replace per pattern, then visually verify each page.

---

## Phase 3: Layout & Polish (1 hr)

### 3.1 Add micro-interactions (hover states, transitions)

**What:** Zero hover effects or transitions on buttons and cards.

**Fix:** Add to `inject_css()`:
```css
.card { transition: box-shadow 0.2s ease, transform 0.2s ease; }
.card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-1px); }
.stButton > button { transition: opacity 0.15s ease; }
.stButton > button:hover { opacity: 0.9; }
```

**Files touched:** `app.py` only.

**Risk:** 🟢 Minimal. CSS-only, no logic change.

---

### 3.2 Fix chat message spacing

**What:** Chat messages currently have default Streamlit spacing.

**Fix:** Add to `inject_css()`:
```css
div[data-testid="stChatMessage"] { margin-bottom: 0.5rem; }
div[data-testid="stChatMessageContent"] { line-height: 1.6; }
```

**Files touched:** `app.py` only.

---

### 3.3 Add responsive breakpoints

**What:** Zero `@media` queries. On smaller screens, the 5-column dashboard nav row wraps awkwardly.

**Fix:** Add to `inject_css()`:
```css
@media (max-width: 768px) {
  .block-container { max-width: 100%; padding: 1rem; }
  .card { padding: 1.25rem; }
}
@media (max-width: 480px) {
  html { font-size: 100%; }
}
```

**Files touched:** `app.py` only.

**Risk:** 🟢 Minimal. CSS-only.

---

### 3.4 Normalize card padding across pages

**What:** `.card` uses `2rem` padding, but module mini-cards use `0.75rem 1rem`. Inconsistent.

**Fix:** Define a `.card-sm` CSS class for compact cards:
```css
.card-sm { padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
```

**Files touched:** `app.py` only.

---

## Phase 4: Advanced (2 hr, optional)

### 4.1 Extract CSS to external file

**What:** All CSS is inside `st.markdown("<style>...</style>")` in `inject_css()`.

**Fix:** Move to `static/style.css` and load via Streamlit's `st.markdown` with file read. Makes CSS editable without touching Python.

**Files touched:** New `static/style.css` + `app.py` `inject_css()`.

---

### 4.2 Add favicon and app icon

**What:** Streamlit default favicon.

**Fix:** Add to `st.set_page_config()`:
```python
st.set_page_config(page_title='AI Mentor', page_icon='🤖', layout='wide')
```

**Files touched:** `app.py` line 1540.

---

### 4.3 Smooth page transitions

**What:** Page changes via sidebar radio are instant with no visual feedback.

**Fix:** Add CSS transition to the main content area:
```css
.main > div { animation: fadeIn 0.2s ease; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
```

**Files touched:** `app.py` only.

---

## Implementation Order (Recommended)

```
Phase 1 (quick fixes)     → 15 min
Phase 2 (theme + classes) → 30 min
Phase 3 (layout + polish) → 1 hr
Phase 4 (advanced)        → 2 hr (optional)
```

Each phase is independent — stop after any phase with no broken dependencies.

---

## Verification Checklist

After each change:
- [ ] App launches without errors
- [ ] Auth page renders correctly
- [ ] Onboarding wizard looks correct (all 3 steps)
- [ ] Dashboard renders with all cards
- [ ] Modules page shows all 13 cards
- [ ] Assessment page lists 15 questions
- [ ] Learning Path page shows roadmap
- [ ] Profile page shows fields
- [ ] AI Mentor chat renders messages
- [ ] All track badge colors display correctly
- [ ] Sidebar navigation works for all pages
- [ ] Sign Out works

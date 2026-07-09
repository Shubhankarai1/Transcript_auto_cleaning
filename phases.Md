# Local and Deployment Setup Phases

## Phase 1: Separate local and live settings
- Remove the need to hardcode the deployed website URL inside the app.
- Keep one setup for local testing and a different setup for the live deployed version.

## Phase 2: Add a local-only environment file
- Create a local env file for development values such as localhost URLs.
- Keep this file out of Git so local settings do not get pushed by mistake.

## Phase 3: Keep a simple shared template
- Add an example env file that shows which values are required.
- This helps the project stay clear without exposing private or local-only settings.

## Phase 4: Make local run automatically use local values
- When the app is started on the laptop, it should pick up local settings by default.
- This allows testing changes safely before pushing anything.

## Phase 5: Keep deployment flow unchanged
- Continue using GitHub Desktop and the normal push process.
- After push, the deployed app should still use live production settings from the hosting platform.

## Expected result
- Local testing works again.
- Deployment continues as usual.
- One codebase supports both local work and the live site without manual switching each time.

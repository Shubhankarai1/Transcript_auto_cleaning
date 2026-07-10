# Database Schema

## Overview
Use Supabase Auth for sign-in and identity. Keep Pinecone as the knowledge layer for transcript chunks and retrieval. For the student-project MVP, the database should stay minimal and only store user profile data, assessment results, and generated roadmaps.

This schema intentionally does not include mentor chat persistence, admin tables, analytics tables, dynamic onboarding-question tables, or extra assessment summary fields that are not required for MVP.

## MVP Tables

### 1. profiles
Purpose: store one onboarding/profile record per authenticated user.

Suggested columns:
- `user_id` uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE
- `email` text
- `full_name` text
- `job_role` text
- `industry` text
- `years_experience` integer
- `career_aspirations` text
- `ai_learning_goals` text
- `weekly_learning_availability` text
- `onboarding_completed` boolean NOT NULL DEFAULT false
- `created_at` timestamptz NOT NULL DEFAULT now()
- `updated_at` timestamptz NOT NULL DEFAULT now()

Why it exists:
- stores the learner's stable background and onboarding answers
- supports profile display and personalization
- gives roadmap generation the learner context it needs

### 2. assessment_attempts
Purpose: store skill-mapping / AI readiness assessment submissions and scoring results.

Suggested columns:
- `id` uuid PRIMARY KEY DEFAULT gen_random_uuid()
- `user_id` uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE
- `raw_answers` jsonb NOT NULL DEFAULT '{}'::jsonb
- `scored_result` jsonb NOT NULL DEFAULT '{}'::jsonb
- `recommended_track` text NOT NULL
- `created_at` timestamptz NOT NULL DEFAULT now()

Why it exists:
- assessment can be taken and retaken
- stores submitted answers as one JSON object for MVP simplicity
- stores deterministic scoring output
- stores the final recommended learner track

Track storage note:
- store the final classification in `recommended_track`
- recommended values should be your product track names, such as:
  - `foundations`
  - `practitioner`
  - `builder`

### 3. roadmap_plans
Purpose: store the personalized roadmap generated for a learner.

Suggested columns:
- `id` uuid PRIMARY KEY DEFAULT gen_random_uuid()
- `user_id` uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE
- `track` text NOT NULL
- `roadmap_json` jsonb NOT NULL DEFAULT '{}'::jsonb
- `version` integer NOT NULL DEFAULT 1
- `is_active` boolean NOT NULL DEFAULT true
- `created_at` timestamptz NOT NULL DEFAULT now()

Why it exists:
- saves the learner's generated study plan
- allows the dashboard to show the current roadmap
- allows regeneration later without losing old versions

## Relationship Summary
- Supabase Auth manages login and provides `auth.users.id`
- `profiles.user_id` links one user to one onboarding/profile record
- `assessment_attempts.user_id` links one user to many assessment attempts
- `roadmap_plans.user_id` links one user to many roadmap versions

## What Is Not Included In MVP
Not included for now:
- `mentor_conversations`
- `mentor_messages`
- `onboarding_questions`
- `learning_tracks`
- `track_modules`
- `track_sessions`
- admin/audit tables
- analytics/progress tracking tables

These can be added later if time remains.

## MVP SQL Example

```sql
create table profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  job_role text,
  industry text,
  years_experience integer,
  career_aspirations text,
  ai_learning_goals text,
  weekly_learning_availability text,
  onboarding_completed boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table assessment_attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  raw_answers jsonb not null default '{}'::jsonb,
  scored_result jsonb not null default '{}'::jsonb,
  recommended_track text not null,
  created_at timestamptz not null default now()
);

create table roadmap_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  track text not null,
  roadmap_json jsonb not null default '{}'::jsonb,
  version integer not null default 1,
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);
```

## RLS Notes
Enable Row Level Security on all three tables.

Recommended MVP policy direction:
- users can read and update only their own `profiles` row
- users can insert and read only their own `assessment_attempts`
- users can insert and read only their own `roadmap_plans`
- backend service role can manage all rows for server-side operations

## Final Recommendation
This is the smallest schema that still matches the actual project flow:
1. user signs in
2. user completes onboarding
3. user takes assessment
4. system computes a recommended track
5. system stores a generated roadmap

Anything smaller than this starts pushing too much product logic into temporary session state or unstructured JSON blobs.


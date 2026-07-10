I'll update database_schema.md with a concise Supabase auth + onboarding schema and SQL examples.

Steps:
1. Replace file content with the minimal schema (two tables).
2. Add sample SQL and RLS notes.
3. Provide a PowerShell command to write the file.gice me  abrief gist of what pinecone backfill is 

Updated file content (will replace Dev Docs/database_schema.md):

# Database Schema

## Overview
Use Supabase built-in `auth` for authentication. Pinecone already holds chunking/embedding vectors; the database stores auth-linked user metadata and onboarding-analysis artifacts. Keep the schema minimal (2 tables): `profiles` and `onboarding_analysis`.

## Tables

### profiles
Purpose: attach app-specific metadata to `auth.users`.

Columns:
- `id` uuid PRIMARY KEY DEFAULT gen_random_uuid()
- `user_id` uuid NOT NULL UNIQUE REFERENCES auth.users(id)
- `email` text
- `full_name` text
- `avatar_url` text
- `role` text
- `metadata` jsonb DEFAULT '{}'::jsonb  — misc preferences
- `created_at` timestamptz DEFAULT now()
- `updated_at` timestamptz DEFAULT now()

Indexes:
- UNIQUE(user_id)
- index on `email`

Notes:
- Lightweight mirror of the auth user fields the UI needs.

### onboarding_analysis
Purpose: track onboarding progress and store analysis/result artifacts per user.

Columns:
- `id` uuid PRIMARY KEY DEFAULT gen_random_uuid()
- `user_id` uuid NOT NULL REFERENCES auth.users(id)
- `status` text NOT NULL DEFAULT 'started'  — values: 'started', 'in_progress', 'completed', 'failed'
- `progress` jsonb DEFAULT '{}'::jsonb  — step flags, timestamps
- `source_transcript_id` text  — optional external ID or pointer to transcript/Pinecone dataset
- `analysis_result` jsonb  — summaries, metrics, flags, RAG metadata
- `consent_given` boolean DEFAULT false
- `created_at` timestamptz DEFAULT now()
- `updated_at` timestamptz DEFAULT now()
- `completed_at` timestamptz NULL

Indexes:
- index on `user_id`
- index on `status`

Notes:
- Store evolving analysis outputs in `jsonb` to avoid frequent migrations.

## Sample SQL (Postgres / Supabase)

Create `profiles`:
```sql
CREATE TABLE profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL UNIQUE REFERENCES auth.users(id),
  email text,
  full_name text,
  avatar_url text,
  role text,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);
CREATE INDEX profiles_email_idx ON profiles (email);
```

Create `onboarding_analysis`:
```sql
CREATE TABLE onboarding_analysis (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id),
  status text NOT NULL DEFAULT 'started',
  progress jsonb DEFAULT '{}'::jsonb,
  source_transcript_id text,
  analysis_result jsonb,
  consent_given boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  completed_at timestamptz
);
CREATE INDEX onboarding_analysis_user_idx ON onboarding_analysis (user_id);
CREATE INDEX onboarding_analysis_status_idx ON onboarding_analysis (status);
```

## RLS / Security
- Enable Row-Level Security on both tables.
  - `profiles`: allow insert for authenticated users; allow select/update where `auth.uid() = user_id`.
  - `onboarding_analysis`: allow users to insert/select/update their own rows; allow admin roles read/write all rows.
- Keep sensitive data minimal; prefer referencing identity fields from `auth.users`.

## Assumptions & Notes
- Pinecone holds embeddings/chunks; DB holds metadata and analysis outputs.
- Two tables keep schema minimal; add `orgs`, `audit_logs`, or `consent_history` later if needed.
- Use `jsonb` for `analysis_result` to avoid frequent schema churn.

Run this PowerShell command from the repo root to overwrite the file now:

```powershell
@"
# Database Schema

## Overview
Use Supabase built-in `auth` for authentication. Pinecone already holds chunking/embedding vectors; the database stores auth-linked user metadata and onboarding-analysis artifacts. Keep the schema minimal (2 tables): `profiles` and `onboarding_analysis`.

## Tables

### profiles
Purpose: attach app-specific metadata to `auth.users`.

Columns:
- `id` uuid PRIMARY KEY DEFAULT gen_random_uuid()
- `user_id` uuid NOT NULL UNIQUE REFERENCES auth.users(id)
- `email` text
- `full_name` text
- `avatar_url` text
- `role` text
- `metadata` jsonb DEFAULT '{}'::jsonb  — misc preferences
- `created_at` timestamptz DEFAULT now()
- `updated_at` timestamptz DEFAULT now()

Indexes:
- UNIQUE(user_id)
- index on `email`

Notes:
- Lightweight mirror of the auth user fields the UI needs.

### onboarding_analysis
Purpose: track onboarding progress and store analysis/result artifacts per user.

Columns:
- `id` uuid PRIMARY KEY DEFAULT gen_random_uuid()
- `user_id` uuid NOT NULL REFERENCES auth.users(id)
- `status` text NOT NULL DEFAULT 'started'  — values: 'started', 'in_progress', 'completed', 'failed'
- `progress` jsonb DEFAULT '{}'::jsonb  — step flags, timestamps
- `source_transcript_id` text  — optional external ID or pointer to transcript/Pinecone dataset
- `analysis_result` jsonb  — summaries, metrics, flags, RAG metadata
- `consent_given` boolean DEFAULT false
- `created_at` timestamptz DEFAULT now()
- `updated_at` timestamptz DEFAULT now()
- `completed_at` timestamptz NULL

Indexes:
- index on `user_id`
- index on `status`

Notes:
- Store evolving analysis outputs in `jsonb` to avoid frequent migrations.

## Sample SQL (Postgres / Supabase)

Create `profiles`:
```sql
CREATE TABLE profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL UNIQUE REFERENCES auth.users(id),
  email text,
  full_name text,
  avatar_url text,
  role text,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);
CREATE INDEX profiles_email_idx ON profiles (email);
```

Create `onboarding_analysis`:
```sql
CREATE TABLE onboarding_analysis (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id),
  status text NOT NULL DEFAULT 'started',
  progress jsonb DEFAULT '{}'::jsonb,
  source_transcript_id text,
  analysis_result jsonb,
  consent_given boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  completed_at timestamptz
);
CREATE INDEX onboarding_analysis_user_idx ON onboarding_analysis (user_id);
CREATE INDEX onboarding_analysis_status_idx ON onboarding_analysis (status);
```

## RLS / Security
- Enable Row-Level Security on both tables.
  - `profiles`: allow insert for authenticated users; allow select/update where `auth.uid() = user_id`.
  - `onboarding_analysis`: allow users to insert/select/update their own rows; allow admin roles read/write all rows.
- Keep sensitive data minimal; prefer referencing identity fields from `auth.users`.

## Assumptions & Notes
- Pinecone holds embeddings/chunks; DB holds metadata and analysis outputs.
- Two tables keep schema minimal; add `orgs`, `audit_logs`, or `consent_history` later if needed.
- Use `jsonb` for `analysis_result` to avoid frequent schema churn.
"@ | Set-Content -Path "Dev Docs\database_schema.md" -Encoding UTF8
```

T
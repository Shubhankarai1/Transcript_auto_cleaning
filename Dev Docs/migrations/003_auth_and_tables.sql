-- Migration 003: Auth tables and RLS policies
-- Run this in Supabase SQL Editor

-- Enable UUID extension (should already be enabled)
create extension if not exists "uuid-ossp";

-- ============================================================
-- Table: assessment_attempts
-- Stores skill-mapping assessment submissions and scores
-- ============================================================
create table if not exists assessment_attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  raw_answers jsonb not null default '{}'::jsonb,
  scored_result jsonb not null default '{}'::jsonb,
  recommended_track text not null,
  created_at timestamptz not null default now()
);

-- ============================================================
-- Table: roadmap_plans
-- Stores generated learning roadmaps per user
-- ============================================================
create table if not exists roadmap_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  track text not null,
  roadmap_json jsonb not null default '{}'::jsonb,
  version integer not null default 1,
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);

-- ============================================================
-- Row Level Security
-- ============================================================
alter table assessment_attempts enable row level security;
alter table roadmap_plans enable row level security;

-- Users can insert and read only their own assessment_attempts
create policy "Users can insert own assessments"
  on assessment_attempts for insert
  with check (auth.uid() = user_id);

create policy "Users can read own assessments"
  on assessment_attempts for select
  using (auth.uid() = user_id);

-- Users can insert and read only their own roadmap_plans
create policy "Users can insert own roadmaps"
  on roadmap_plans for insert
  with check (auth.uid() = user_id);

create policy "Users can read own roadmaps"
  on roadmap_plans for select
  using (auth.uid() = user_id);

-- Service role can manage all rows (backend uses service role)
-- This is automatic with service_role key, no policy needed.

-- ============================================================
-- profiles table already exists from earlier migration
-- Add RLS if not already present
-- ============================================================
alter table profiles enable row level security;

create policy "Users can read own profile"
  on profiles for select
  using (auth.uid() = user_id);

create policy "Users can insert own profile"
  on profiles for insert
  with check (auth.uid() = user_id);

create policy "Users can update own profile"
  on profiles for update
  using (auth.uid() = user_id);

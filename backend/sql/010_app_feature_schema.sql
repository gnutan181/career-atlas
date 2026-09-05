-- 10. Remaining persistence tables for gap analysis, GitHub analysis, jobs,
--     milestones, and deep research. Safe to run after 009_resume_profile_schema.sql.

CREATE TABLE IF NOT EXISTS public.skill_gaps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id UUID REFERENCES public.resumes(id) ON DELETE CASCADE,
    target_role TEXT,
    skill TEXT,
    category TEXT,
    relevance INTEGER,
    difficulty TEXT,
    level_required TEXT,
    prerequisites TEXT[] NOT NULL DEFAULT '{}',
    why TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE public.skill_gaps ADD COLUMN IF NOT EXISTS resume_id UUID REFERENCES public.resumes(id) ON DELETE CASCADE;
ALTER TABLE public.skill_gaps ADD COLUMN IF NOT EXISTS target_role TEXT;
ALTER TABLE public.skill_gaps ADD COLUMN IF NOT EXISTS level_required TEXT;
ALTER TABLE public.skill_gaps ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'skill_gaps' AND column_name = 'user_id') THEN
        ALTER TABLE public.skill_gaps ALTER COLUMN user_id DROP NOT NULL;
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS skill_gaps_resume_role_created_idx ON public.skill_gaps (resume_id, target_role, created_at DESC);

CREATE TABLE IF NOT EXISTS public.github_tokens (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at TIMESTAMPTZ,
    github_user_id TEXT,
    github_username TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS public.github_profiles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    analysis_summary TEXT,
    coding_behavior TEXT,
    inferred_skills TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS public.github_repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    repo_name TEXT NOT NULL,
    repo_url TEXT,
    is_owner BOOLEAN NOT NULL DEFAULT TRUE,
    description TEXT,
    primary_language TEXT,
    analysis_summary TEXT,
    coding_behavior TEXT,
    languages JSONB,
    commit_count INTEGER,
    first_commit_at TIMESTAMPTZ,
    last_commit_at TIMESTAMPTZ,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, repo_name)
);
ALTER TABLE public.github_repositories ADD COLUMN IF NOT EXISTS languages JSONB;
ALTER TABLE public.github_repositories ADD COLUMN IF NOT EXISTS commit_count INTEGER;
ALTER TABLE public.github_repositories ADD COLUMN IF NOT EXISTS first_commit_at TIMESTAMPTZ;
ALTER TABLE public.github_repositories ADD COLUMN IF NOT EXISTS last_commit_at TIMESTAMPTZ;
CREATE TABLE IF NOT EXISTS public.github_skill_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    skill TEXT NOT NULL,
    evidence TEXT,
    evidence_type TEXT NOT NULL DEFAULT 'file' CHECK (evidence_type IN ('file', 'language', 'commit')),
    confidence TEXT NOT NULL DEFAULT 'low',
    source_repo TEXT,
    confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, skill, source_repo)
);
ALTER TABLE public.github_skill_evidence ADD COLUMN IF NOT EXISTS evidence_type TEXT NOT NULL DEFAULT 'file';
ALTER TABLE public.github_skill_evidence DROP CONSTRAINT IF EXISTS github_skill_evidence_type_check;
ALTER TABLE public.github_skill_evidence
    ADD CONSTRAINT github_skill_evidence_type_check
    CHECK (evidence_type IN ('file', 'language', 'commit'));
CREATE INDEX IF NOT EXISTS github_skill_evidence_user_confirmed_idx ON public.github_skill_evidence (user_id, confirmed);

CREATE TABLE IF NOT EXISTS public.job_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    job_id TEXT,
    query_role TEXT,
    user_location_preference TEXT,
    title TEXT,
    company TEXT,
    location TEXT,
    remote BOOLEAN NOT NULL DEFAULT FALSE,
    seniority TEXT,
    match_pct INTEGER NOT NULL DEFAULT 0,
    matched TEXT[] NOT NULL DEFAULT '{}',
    missing TEXT[] NOT NULL DEFAULT '{}',
    salary TEXT,
    posted_days INTEGER NOT NULL DEFAULT 0,
    description TEXT,
    external_url TEXT,
    score_json JSONB,
    explanation_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE public.job_matches ADD COLUMN IF NOT EXISTS job_id TEXT;
ALTER TABLE public.job_matches ADD COLUMN IF NOT EXISTS query_role TEXT;
ALTER TABLE public.job_matches ADD COLUMN IF NOT EXISTS user_location_preference TEXT;
ALTER TABLE public.job_matches ADD COLUMN IF NOT EXISTS score_json JSONB;
ALTER TABLE public.job_matches ADD COLUMN IF NOT EXISTS explanation_json JSONB;
CREATE INDEX IF NOT EXISTS job_matches_user_match_idx ON public.job_matches (user_id, match_pct DESC);

CREATE TABLE IF NOT EXISTS public.milestones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    target_role_id TEXT,
    resume_id UUID REFERENCES public.resumes(id) ON DELETE SET NULL,
    phase TEXT,
    title TEXT,
    skill TEXT,
    status TEXT NOT NULL DEFAULT 'locked',
    estimated_weeks INTEGER,
    description TEXT,
    courses JSONB NOT NULL DEFAULT '[]'::jsonb,
    project JSONB NOT NULL DEFAULT '{}'::jsonb,
    checklist TEXT[] NOT NULL DEFAULT '{}',
    sort_order INTEGER NOT NULL DEFAULT 0,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE public.milestones ADD COLUMN IF NOT EXISTS target_role_id TEXT;
ALTER TABLE public.milestones ADD COLUMN IF NOT EXISTS resume_id UUID REFERENCES public.resumes(id) ON DELETE SET NULL;
ALTER TABLE public.milestones ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS milestones_user_role_idx ON public.milestones (user_id, target_role_id, sort_order);

CREATE TABLE IF NOT EXISTS public.learning_pathways (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    target_role_id TEXT,
    role_slug TEXT NOT NULL,
    title TEXT,
    estimated_weeks INTEGER NOT NULL DEFAULT 0,
    milestones JSONB NOT NULL DEFAULT '[]'::jsonb,
    resources JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE public.learning_pathways ADD COLUMN IF NOT EXISTS target_role_id TEXT;
ALTER TABLE public.learning_pathways ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE public.learning_pathways ADD COLUMN IF NOT EXISTS estimated_weeks INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.learning_pathways ADD COLUMN IF NOT EXISTS milestones JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE public.learning_pathways ADD COLUMN IF NOT EXISTS resources JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE public.learning_pathways ADD COLUMN IF NOT EXISTS description TEXT;
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'learning_pathways' AND column_name = 'target_role') THEN
        ALTER TABLE public.learning_pathways ALTER COLUMN target_role DROP NOT NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'learning_pathways' AND column_name = 'pathway') THEN
        ALTER TABLE public.learning_pathways ALTER COLUMN pathway DROP NOT NULL;
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS learning_pathways_user_role_idx ON public.learning_pathways (user_id, role_slug, created_at DESC);

ALTER TABLE public.skill_gaps ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.github_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.github_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.github_repositories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.github_skill_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.job_matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.milestones ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.learning_pathways ENABLE ROW LEVEL SECURITY;

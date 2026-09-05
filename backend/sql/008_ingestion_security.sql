-- 8. Evidence provenance and explicit confirmation gate
-- Apply after 007_github_hardening.sql.

ALTER TABLE github_skill_evidence
    ADD COLUMN IF NOT EXISTS evidence_type TEXT NOT NULL DEFAULT 'file';

ALTER TABLE github_skill_evidence
    DROP CONSTRAINT IF EXISTS github_skill_evidence_type_check;
ALTER TABLE github_skill_evidence
    ADD CONSTRAINT github_skill_evidence_type_check
    CHECK (evidence_type IN ('file', 'language', 'commit'));

-- Existing historical suggestions were created before the explicit-review
-- requirement. Requiring confirmation prevents them from silently influencing
-- skill gaps or job matches after this migration.
UPDATE github_skill_evidence SET confirmed = FALSE;

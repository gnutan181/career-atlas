# Backend (FastAPI)

## Run
```bash
uv sync
uv run uvicorn app.main:app --reload
```

## API Routes
- `POST /api/parse-resume/`
  - accepts multipart `file` (preferred) or `resume_key` form field
  - extracts resume via Gemini
  - writes normalized resume data into Supabase tables
- `POST /api/parse-resume/manual`
  - accepts structured `ResumeExtraction` JSON body
  - writes manually entered profile data into Supabase tables
- `GET /api/parse-resume/latest`
  - returns the latest fully assembled resume payload
- `GET /api/parse-resume/all`
  - returns a list of all resumes/profiles owned by the authenticated user
- `POST /api/parse-resume/select/{resume_id}`
  - selects a resume/profile as active by updating its `created_at` timestamp
- `POST /api/analyze-gaps/`
  - body: `{ "target_role_title": "..." }`
- `POST /api/generate-roadmap/`
  - body: `{ "target_role_id": "..." }`
  - if `milestones` table is missing, returns generated milestones with `persisted=false`
- `POST /api/research-jobs/`
  - ranks live Adzuna postings against the user's confirmed skills (resume  confirmed GitHub)

### GitHub profile analysis
- `POST /api/github/oauth/callback`  exchanges the OAuth `code` for a token (CSRF `state` verified client-side), stores it
- `GET  /api/github/repos`  lists the user's owned + contributed (non-fork) repos
- `POST /api/github/analyze`  analyzes selected repos; writes per-repo facts + evidence-bound skills (high-confidence auto-confirmed, rest quarantined)
- `GET  /api/github/profile`  stored profile + per-repo facts + skill evidence (for the insights panel)
- `POST /api/github/skills/confirm` and `/api/github/skills/reject`  promote/discard suggested skills (body: `{ "evidence_ids": [...] }`)
- `GET  /api/github/status`  whether GitHub is connected (+ username)

## Retrieval release gate

`uv run python scripts/run_retrieval_benchmark.py` evaluates the deterministic
120-case curated-role suite. A release passes only at **Recall@10 ≥ 0.80** and
**nDCG@10 ≥ 0.72**. Run it against the production-equivalent Pinecone, embedding,
and reranker configuration before changing retrieval models, taxonomy, or weights.

## Ingestion safeguards

Resume upload accepts a verified PDF up to 10 MB. Direct identifiers (see
**Guardrails** below) are replaced with temporary tokens before the extraction
provider sees the text; only the candidate's own name, email and phone are
restored, and only in validated output. Resume and repository
contents are explicitly isolated as untrusted data in prompts. API calls are
rate-limited per bearer token (per IP before authentication); use a shared Redis
store for the same policy when deploying more than one backend instance.

## Guardrails

Three layers sit around every LLM call that touches untrusted text
(`app/security/`):

| Layer | What it does | Switch |
|---|---|---|
| **PII masking** (`pii.py`, Microsoft Presidio) | Before any LLM call, names, emails, phones (Indian + international), street addresses / PIN codes, PAN, Aadhaar (Verhoeff-checked), passport and voter IDs become tokens like `[[CA_NAME_1]]`. Companies, universities, titles, skills, dates and cities are never masked. The original email/phone regex runs afterwards as a safety net and is the fallback if Presidio fails. | `PII_ENGINE`, `PII_ENTITIES`, `PII_SCORE_THRESHOLD`, `PRESIDIO_SPACY_MODEL` |
| **Prompt Shields** (`prompt_shields.py`, Azure AI Content Safety) | Document-attack detection on the *masked* resume and on GitHub files/descriptions, in ≤2,000-char overlapping chunks, 5 per request (API limit: 5 docs / 10K chars, `api-version=2024-09-01`). Resume flagged → HTTP 422; shield error/timeout → HTTP 503 (fail closed). GitHub file flagged or unchecked → skipped and listed in `skipped_files`. `isolate_untrusted_text()` stays as a second layer. | `PROMPT_SHIELDS_ENABLED` (off by default) |
| **Output guards** (`output_guards.py`, Guardrails AI) | Profiles: `ResumeExtraction` schema + only the candidate's own name/email/phone may appear (other masked values are never restored). Gap analysis and roadmaps: `ProfanityFree`, optional `ToxicLanguage`, and a custom check that rejects "verified/proven/demonstrated <skill>" when the skill is not on the resume or confirmed on GitHub. Failure → one re-ask → HTTP 502 with a safe message. Guardrails Hub telemetry is disabled in code. | `OUTPUT_GUARDS_ENABLED`, `TOXICITY_GUARD_ENABLED` |

Each check writes one content-free log line on the `app.guardrails` logger
(`check`, `outcome`, `latency_ms`, counts).

**Setup.** `uv sync` (or `pip install -r requirements.txt`, as the Dockerfile does)
also installs the spaCy model Presidio uses (`en_core_web_lg` 3.8.0, about 445 MB installed, pinned
by URL). Without the model, masking falls back to the regex and logs a warning.
The toxicity validator is optional and heavy (torch):
`uv pip install guardrails-ai-toxic-language && python -m guardrails_ai.toxic_language.post_install`, then `TOXICITY_GUARD_ENABLED=true`.

**LLM provider.** `LLM_PROVIDER=default` keeps the existing Gemini/Groq split.
`azure` sends every LLM call to an Azure AI Foundry deployment through its
OpenAI-compatible v1 endpoint (`AZURE_AI_ENDPOINT=https://<resource>.services.ai.azure.com/openai/v1`,
`AZURE_AI_API_KEY`, `AZURE_AI_DEPLOYMENT`); `groq` / `gemini` force one provider.

**Evaluation.** `python eval/guardrails/run_eval.py` (synthetic data only) reports PII
recall/precision per entity, facts kept, output-guard false blocks and latency;
`--extraction` adds a live LLM before/after-Presidio extraction comparison, and
Prompt Shields rates are measured only when Azure Content Safety is configured.

## Environment
See **`backend/.env.example`** for the authoritative, copy-pasteable list. Key vars:
- `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SECRET_KEY` (service-role; bypasses RLS)
- `GOOGLE_API_KEY`  primary Gemini key
- `GOOGLE_API_KEY_1`  `GOOGLE_API_KEY_4`  optional extra keys for rotation (14; falls back to the single key)
- `GROQ_API_KEY`, `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)
- `TAVILY_API_KEY`, `JINA_API_KEY`
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_REGION`, `PINECONE_HOST`
- `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`  job search
- `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`  GitHub OAuth App (repo analysis)
- `SENTRY_DSN`  optional error tracking
- `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT`  optional tracing
- `CORS_ORIGINS=http://localhost:8080,http://localhost:5173,http://localhost:3000`

> **Prod note:** the deployed backend reads these from **AWS Secrets Manager** (injected into Lambda), *not* from `.env`. Editing `.env` only affects local dev.

Optional dev bypass:
- `DEV_BYPASS_AUTH=true`
- `DEV_USER_ID=<uuid from supabase auth.users>`

## Resume Storage
- Uploaded resume PDFs are stored in the **Supabase Storage** bucket named `resumes` (private). See `RESUME_BUCKET` in `app/utils/storage.py`.

## Expected Supabase Tables (current schema integration)
- `resumes`
- `contacts`
- `skills`
- `programming_languages`
- `spoken_languages`
- `keywords`
- `experiences`
- `experience_bullets`
- `experience_technologies`
- `education`
- `education_notes`
- `projects`
- `project_technologies`
- `certifications`

Optional but used by downstream flows:
- `skill_gaps`
- `milestones`
- `target_roles`
- `job_matches`
- `profiles`

GitHub profile analysis (see `backend/sql/006_*` and `007_*`):
- `github_tokens`  stored OAuth access token (RLS, cascade on user delete)
- `github_profiles`  overall summary + coding behavior + inferred skill names
- `github_repositories`  per-repo summary, languages %, commit stats
- `github_skill_evidence`  quarantined inferred skills (skill, evidence, confidence, source_repo, `confirmed` bool); only `confirmed=true` rows feed the profile/gaps/jobs

`job_matches` now also stores the structured job-search payload when the schema migration is applied:
- `job_id`
- `query_role`
- `user_location_preference`
- `score_json`
- `explanation_json`

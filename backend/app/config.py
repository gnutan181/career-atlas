from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

def _get_google_api_keys() -> list[str]:
    keys = []
    main_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if main_key:
        keys.append(main_key)
    for i in range(1, 5):
        k = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Deliberately fail closed: if ENVIRONMENT is unset (e.g. a deployment that
    # forgot to set it), default to "production" so the dev auth bypass in
    # app.dependencies.auth stays disabled. Local dev gets ENVIRONMENT=development
    # from backend/.env.example.
    # Keep this as a literal default rather than evaluating ``os.getenv`` at
    # import time. BaseSettings still reads ENVIRONMENT and backend/.env when
    # Settings is instantiated, while a genuinely unset environment safely
    # falls back to production.
    environment: str = "production"
    cors_origins: str = os.getenv("CORS_ORIGINS", "https://career-atlas.gnutan181.workers.dev, http://localhost:3000,http://localhost:5173,http://localhost:8080")
    api_rate_limit_requests: int = int(os.getenv("API_RATE_LIMIT_REQUESTS", "30"))
    api_rate_limit_window_seconds: int = int(os.getenv("API_RATE_LIMIT_WINDOW_SECONDS", "60"))
    
    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "https://placeholder.supabase.co")
    # New-style keys (preferred): sb_publishable_* / sb_secret_*
    # Old-style fallback: anon JWT / service_role JWT (SUPABASE_SERVICE_KEY)
    supabase_publishable_key: str = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
    supabase_secret_key: str = os.getenv("SUPABASE_SECRET_KEY", os.getenv("SUPABASE_SERVICE_KEY", ""))
    # Kept for back-compat with code that still reads the old name.
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_SECRET_KEY", ""))
    supabase_jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET", "")
    supabase_jwt_public_key: str = os.getenv("SUPABASE_JWT_PUBLIC_KEY", "")
    
    # LLMs
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_api_keys: list[str] = _get_google_api_keys()
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gap_analysis_model: str = os.getenv("GAP_ANALYSIS_MODEL", "gemini-1.5-flash")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # Tools
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Monitoring
    sentry_dsn: str = os.getenv("SENTRY_DSN", "")
    adzuna_app_id: str = os.getenv("ADZUNA_APP_ID", "")
    adzuna_app_key: str = os.getenv("ADZUNA_APP_KEY", "")

    # Jina & Pinecone (Gap Analysis)
    jina_api_key: str = os.getenv("JINA_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "careeratlas")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_host: str = os.getenv("PINECONE_HOST", "")

    # Dev convenience: allows backend routes to run without bearer token.
    dev_bypass_auth: bool = os.getenv("DEV_BYPASS_AUTH", "false").lower() == "true"
    dev_user_id: str = os.getenv("DEV_USER_ID", "")

    # GitHub OAuth
    github_client_id: str = os.getenv("GITHUB_CLIENT_ID", "")
    github_client_secret: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    github_oauth_redirect_uri: str = os.getenv("GITHUB_OAUTH_REDIRECT_URI", "")

    # ── LLM provider routing ─────────────────────────────────────────────
    # "default" keeps the historical mix (Gemini for extraction / gap analysis,
    # Groq for structured chains).  "azure" routes every helper in
    # app.utils.llm_factory to an Azure AI Foundry deployment; "groq" and
    # "gemini" force a single provider.
    llm_provider: str = "default"
    # Foundry OpenAI-compatible v1 endpoint, e.g.
    # https://<resource>.services.ai.azure.com/openai/v1
    azure_ai_endpoint: str = ""
    azure_ai_api_key: str = ""
    azure_ai_deployment: str = ""
    azure_ai_timeout_seconds: float = 60.0

    # ── Guardrails: PII masking (Presidio) ───────────────────────────────
    pii_engine: str = "presidio"  # presidio | regex
    # Never list LOCATION / ORGANIZATION / DATE_TIME / NRP: the extractor needs
    # cities, companies, universities and dates.
    pii_entities: str = (
        "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,STREET_ADDRESS,"
        "IN_PAN,IN_AADHAAR,IN_PASSPORT,IN_VOTER"
    )
    pii_score_threshold: float = 0.5
    presidio_spacy_model: str = "en_core_web_lg"

    # ── Guardrails: Azure AI Content Safety Prompt Shields ───────────────
    prompt_shields_enabled: bool = False
    azure_content_safety_endpoint: str = ""
    azure_content_safety_key: str = ""
    prompt_shields_timeout_seconds: float = 10.0

    # ── Guardrails: output validation (Guardrails AI) ────────────────────
    output_guards_enabled: bool = True
    # toxic_language pulls in torch + a local BERT model; off unless installed.
    toxicity_guard_enabled: bool = False

    @property
    def pii_entity_list(self) -> list[str]:
        return [e.strip().upper() for e in self.pii_entities.split(",") if e.strip()]

settings = Settings()

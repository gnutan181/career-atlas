"""Shared resume-lookup helpers.

Several routers need "the user's resumes, newest first" or just "the latest
one". These lookups are always strictly scoped to `user_id` — never a global
fallback, which would hand one user another user's resume.
"""
from app.dependencies.database import db_client


def user_resume_ids(user_id: str) -> list[str]:
    """All of the user's resume ids, newest first. Strictly scoped to user_id."""
    try:
        resp = (
            db_client.table("resumes")
            .select("id")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return [r["id"] for r in (resp.data or []) if r.get("id")]
    except Exception:
        return []


def latest_resume_id(user_id: str) -> str | None:
    """The user's most recent resume id, or None if they have none."""
    ids = user_resume_ids(user_id)
    return ids[0] if ids else None


def evidence_skills(user_id: str, resume_id: str | None) -> list[str]:
    """Skills the user has evidence for: the resume's skills and programming
    languages plus CONFIRMED GitHub skills (never quarantined guesses).
    Used by the output guards to reject unsupported "verified" claims."""
    skills: list[str] = []
    try:
        if resume_id:
            for table, column in (("skills", "skill"), ("programming_languages", "language")):
                resp = db_client.table(table).select(column).eq("resume_id", resume_id).execute()
                skills.extend(row[column] for row in (resp.data or []) if row.get(column))
        confirmed = (
            db_client.table("github_skill_evidence")
            .select("skill")
            .eq("user_id", user_id)
            .eq("confirmed", True)
            .execute()
        )
        skills.extend(row["skill"] for row in (confirmed.data or []) if row.get("skill"))
    except Exception:
        return skills
    return list(dict.fromkeys(skills))

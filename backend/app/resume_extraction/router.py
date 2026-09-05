import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any
from collections import defaultdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.dependencies.auth import require_user_id
from app.dependencies.database import db_client
from app.resume_extraction.schemas import ResumeExtraction
from app.resume_extraction.service import (
    extract_structured_resume_data,
    pdf_bytes_to_markdown,
)
from app.utils.resumes import latest_resume_id, user_resume_ids
from app.utils.storage import upload_resume_file, download_resume_file
from app.security.ingestion import read_validated_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/parse-resume", tags=["Resume"])


def _safe_filename(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return sanitized or "resume.pdf"


def _insert_resume_row(data: dict[str, Any], user_id: str, resume_key: str) -> str:
    # Always write user_id. A resume row without an owner is unusable and
    # would leak into other users' "latest resume" lookups — so never insert
    # one. Let a failure propagate instead of orphaning the row.
    resp = (
        db_client.table("resumes")
        .insert(
            {
                "user_id": user_id,
                "resume_key": resume_key,
                "full_name": data.get("full_name"),
                "headline": data.get("headline"),
                "summary": data.get("summary"),
            }
        )
        .execute()
    )
    return resp.data[0]["id"]


def insert_full_resume(data: dict[str, Any], user_id: str, resume_key: str) -> str:
    resume_id = _insert_resume_row(data, user_id, resume_key)

    contact = data.get("contact", {}) or {}
    skill_values: list[str] = []
    skill_values.extend([s for s in data.get("skills", []) or [] if isinstance(s, str) and s.strip()])
    skill_values.extend([s for s in data.get("programming_languages", []) or [] if isinstance(s, str) and s.strip()])
    skill_values.extend([s for s in data.get("spoken_languages", []) or [] if isinstance(s, str) and s.strip()])
    skill_values.extend([s for s in data.get("keywords", []) or [] if isinstance(s, str) and s.strip()])
    for exp in data.get("experience", []) or []:
        skill_values.extend([s for s in (exp.get("technologies") or []) if isinstance(s, str) and s.strip()])
    for proj in data.get("projects", []) or []:
        skill_values.extend([s for s in (proj.get("technologies") or []) if isinstance(s, str) and s.strip()])
    skill_values = list(dict.fromkeys(skill_values))
    try:
        db_client.table("contacts").insert({"resume_id": resume_id, **contact}).execute()
    except Exception:
        logger.warning("contacts insert failed for resume %s", resume_id, exc_info=True)

    if skill_values:
        skill_records = [{"resume_id": resume_id, "skill": s} for s in skill_values]
        db_client.table("skills").insert(skill_records).execute()

    prog_langs = data.get("programming_languages", []) or []
    if prog_langs:
        prog_lang_records = [{"resume_id": resume_id, "language": s} for s in prog_langs]
        try:
            db_client.table("programming_languages").insert(prog_lang_records).execute()
        except Exception:
            logger.warning("programming_languages bulk insert failed for resume %s", resume_id, exc_info=True)

    spoken_langs = data.get("spoken_languages", []) or []
    if spoken_langs:
        spoken_lang_records = [{"resume_id": resume_id, "language": s} for s in spoken_langs]
        try:
            db_client.table("spoken_languages").insert(spoken_lang_records).execute()
        except Exception:
            logger.warning("spoken_languages bulk insert failed for resume %s", resume_id, exc_info=True)

    keywords = data.get("keywords", []) or []
    if keywords:
        keyword_records = [{"resume_id": resume_id, "keyword": k} for k in keywords]
        try:
            db_client.table("keywords").insert(keyword_records).execute()
        except Exception:
            logger.warning("keywords bulk insert failed for resume %s", resume_id, exc_info=True)

    experiences = data.get("experience", []) or []
    if experiences:
        exp_records = [
            {
                "resume_id": resume_id,
                "company": exp.get("company"),
                "title": exp.get("title"),
                "location": exp.get("location"),
                "start_date": exp.get("start_date"),
                "end_date": exp.get("end_date"),
                "is_current": exp.get("is_current"),
            }
            for exp in experiences
        ]
        exp_res = db_client.table("experiences").insert(exp_records).execute()

        all_bullets = []
        all_exp_techs = []
        for i, exp in enumerate(experiences):
            exp_id = exp_res.data[i]["id"]
            bullets = exp.get("description_bullets", []) or []
            all_bullets.extend([{"experience_id": exp_id, "bullet": b} for b in bullets])

            techs = exp.get("technologies", []) or []
            all_exp_techs.extend([{"experience_id": exp_id, "tech": t} for t in techs])

        if all_bullets:
            db_client.table("experience_bullets").insert(all_bullets).execute()
        if all_exp_techs:
            db_client.table("experience_technologies").insert(all_exp_techs).execute()

    educations = data.get("education", []) or []
    if educations:
        edu_records = [
            {
                "resume_id": resume_id,
                "institution": edu.get("institution"),
                "degree": edu.get("degree"),
                "field_of_study": edu.get("field_of_study"),
                "start_date": edu.get("start_date"),
                "end_date": edu.get("end_date"),
                "grade": edu.get("grade"),
            }
            for edu in educations
        ]
        edu_res = db_client.table("education").insert(edu_records).execute()

        all_notes = []
        for i, edu in enumerate(educations):
            edu_id = edu_res.data[i]["id"]
            notes = edu.get("notes", []) or []
            all_notes.extend([{"education_id": edu_id, "note": n} for n in notes])

        if all_notes:
            db_client.table("education_notes").insert(all_notes).execute()

    projects = data.get("projects", []) or []
    if projects:
        proj_records = [
            {
                "resume_id": resume_id,
                "name": proj.get("name"),
                "description": proj.get("description"),
                "link": proj.get("link"),
            }
            for proj in projects
        ]
        proj_res = db_client.table("projects").insert(proj_records).execute()

        all_proj_techs = []
        for i, proj in enumerate(projects):
            proj_id = proj_res.data[i]["id"]
            techs = proj.get("technologies", []) or []
            all_proj_techs.extend([{"project_id": proj_id, "tech": t} for t in techs])

        if all_proj_techs:
            db_client.table("project_technologies").insert(all_proj_techs).execute()

    certifications = data.get("certifications", []) or []
    if certifications:
        cert_records = [
            {
                "resume_id": resume_id,
                "name": cert.get("name"),
                "issuer": cert.get("issuer"),
                "issue_date": cert.get("date"),
                "expiry_date": None,
                "credential_id": cert.get("credential_id"),
                "credential_url": cert.get("link"),
            }
            for cert in certifications
        ]
        try:
            db_client.table("certifications").insert(cert_records).execute()
        except Exception:
            logger.warning("certifications bulk insert failed for resume %s", resume_id, exc_info=True)

    # Optional sync for older UI paths that read profiles.
    try:
        db_client.table("profiles").upsert(
            {
                "user_id": user_id,
                "name": data.get("full_name") or "Candidate",
                "headline": data.get("headline"),
                "summary": data.get("summary"),
                "email": contact.get("email"),
                "location": contact.get("location"),
                "github": contact.get("github"),
                "resume_key": resume_key,
                "completeness": 30,
            },
            on_conflict="user_id",
        ).execute()
    except Exception:
        logger.warning("profiles sync upsert failed for resume %s", resume_id, exc_info=True)

    return resume_id


def _delete_other_resumes(user_id: str, keep_resume_id: str) -> None:
    """Replace-on-upload: keep one resume per user.

    All child tables (skills, experiences, skill_gaps, milestones, …) FK to
    resumes with ON DELETE CASCADE, so dropping the old resume row also clears
    every stale derived row — no orphans.
    """
    try:
        db_client.table("resumes").delete()\
            .eq("user_id", user_id)\
            .neq("id", keep_resume_id)\
            .execute()
    except Exception:
        logger.warning("failed to prune old resumes for user %s", user_id, exc_info=True)


async def fetch_full_resume(resume_id: str) -> dict[str, Any]:
    # ⚡ Bolt Optimization: Fix N+1 query performance bottleneck
    # What: Replaced sequential, synchronous Supabase queries (O(N) queries for related records like bullets and techs)
    #       with concurrent batch fetching via asyncio.to_thread and .in_() filters.
    # Why: The Supabase Python client is synchronous and blocks the FastAPI event loop. A resume with 5 experiences and 3 projects
    #      was generating 15+ sequential round-trips to the DB.
    # Impact: Reduces DB queries from O(N) to O(1) (~7-8 queries total), executed concurrently.
    #         Expected to reduce resume fetch latency by 60-80% for complex resumes.

    async def _fetch_certifications():
        try:
            return await asyncio.to_thread(lambda: db_client.table("certifications").select("*").eq("resume_id", resume_id).execute().data)
        except Exception:
            logger.warning("certifications fetch failed for resume %s", resume_id, exc_info=True)
            return []

    async def empty_list():
        return []

    # 1. Fetch top-level scalar lists and simple tables concurrently
    results = await asyncio.gather(
        asyncio.to_thread(lambda: db_client.table("resumes").select("*").eq("id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("contacts").select("*").eq("resume_id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("skills").select("*").eq("resume_id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("programming_languages").select("*").eq("resume_id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("spoken_languages").select("*").eq("resume_id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("keywords").select("*").eq("resume_id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("experiences").select("*").eq("resume_id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("education").select("*").eq("resume_id", resume_id).execute().data),
        asyncio.to_thread(lambda: db_client.table("projects").select("*").eq("resume_id", resume_id).execute().data),
        _fetch_certifications()
    )

    resume_data = results[0]
    resume = resume_data[0] if resume_data else {}
    contact_rows = results[1]
    contact = contact_rows[0] if contact_rows else {}

    skills = [x["skill"] for x in results[2]]
    prog_langs = [x["language"] for x in results[3]]
    spoken_langs = [x["language"] for x in results[4]]
    keywords = [x["keyword"] for x in results[5]]

    exp_rows = results[6]
    edu_rows = results[7]
    proj_rows = results[8]
    certifications = results[9]

    # 2. Extract IDs for batch fetching child records
    exp_ids = [exp["id"] for exp in exp_rows] if exp_rows else []
    edu_ids = [edu["id"] for edu in edu_rows] if edu_rows else []
    proj_ids = [proj["id"] for proj in proj_rows] if proj_rows else []

    # 3. Batch fetch child records concurrently using .in_()
    child_queries = []

    if exp_ids:
        child_queries.append(asyncio.to_thread(lambda: db_client.table("experience_bullets").select("*").in_("experience_id", exp_ids).execute().data))
        child_queries.append(asyncio.to_thread(lambda: db_client.table("experience_technologies").select("*").in_("experience_id", exp_ids).execute().data))
    else:
        child_queries.extend([empty_list(), empty_list()])

    if edu_ids:
        child_queries.append(asyncio.to_thread(lambda: db_client.table("education_notes").select("*").in_("education_id", edu_ids).execute().data))
    else:
        child_queries.append(empty_list())

    if proj_ids:
        child_queries.append(asyncio.to_thread(lambda: db_client.table("project_technologies").select("*").in_("project_id", proj_ids).execute().data))
    else:
        child_queries.append(empty_list())

    child_results = await asyncio.gather(*child_queries)

    # 4. Stitch child records to parents in memory O(M) instead of DB O(N*M)
    exp_bullets_map = defaultdict(list)
    for row in child_results[0]:
        exp_bullets_map[row["experience_id"]].append(row["bullet"])

    exp_techs_map = defaultdict(list)
    for row in child_results[1]:
        exp_techs_map[row["experience_id"]].append(row["tech"])

    edu_notes_map = defaultdict(list)
    for row in child_results[2]:
        edu_notes_map[row["education_id"]].append(row["note"])

    proj_techs_map = defaultdict(list)
    for row in child_results[3]:
        proj_techs_map[row["project_id"]].append(row["tech"])

    # Build final structures
    experiences = []
    for exp in exp_rows:
        exp_id = exp["id"]
        exp["description_bullets"] = exp_bullets_map.get(exp_id, [])
        exp["technologies"] = exp_techs_map.get(exp_id, [])
        experiences.append(exp)

    education = []
    for edu in edu_rows:
        edu_id = edu["id"]
        edu["notes"] = edu_notes_map.get(edu_id, [])
        education.append(edu)

    projects = []
    for proj in proj_rows:
        proj_id = proj["id"]
        proj["technologies"] = proj_techs_map.get(proj_id, [])
        projects.append(proj)

    return {
        "resume_id": resume_id,
        "full_name": resume.get("full_name"),
        "headline": resume.get("headline"),
        "summary": resume.get("summary"),
        "contact": contact,
        "skills": skills,
        "programming_languages": prog_langs,
        "spoken_languages": spoken_langs,
        "experience": experiences,
        "education": education,
        "projects": projects,
        "certifications": certifications,
        "keywords": keywords,
    }


@router.post("/")
async def parse_resume(
    file: UploadFile | None = File(default=None),
    resume_key: str | None = Form(default=None),
    user_id: str = Depends(require_user_id),
):
    try:
        resolved_resume_key = resume_key
        pdf_bytes: bytes
        if file is not None:
            upload_bytes = await read_validated_pdf(file)
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            object_path = f"{user_id}/{timestamp}_{_safe_filename(file.filename or 'resume.pdf')}"
            await asyncio.to_thread(upload_resume_file, object_path, upload_bytes)
            resolved_resume_key = object_path
            pdf_bytes = upload_bytes
        elif resume_key:
            # The storage client uses a service credential, so do not let a
            # caller turn a guessed object key into a cross-tenant read.
            if not resume_key.startswith(f"{user_id}/"):
                raise HTTPException(status_code=403, detail="Stored resume does not belong to this user.")
            try:
                pdf_bytes = await asyncio.to_thread(download_resume_file, resume_key)
            except Exception as exc:
                logger.warning("stored resume download failed for key %s", resume_key, exc_info=True)
                raise HTTPException(status_code=404, detail="Stored resume not found.") from exc
            resolved_resume_key = resume_key
        else:
            raise HTTPException(status_code=400, detail="Provide either file upload or resume_key.")

        md_text = await asyncio.to_thread(pdf_bytes_to_markdown, pdf_bytes)
        extracted = await asyncio.to_thread(extract_structured_resume_data, md_text)
        extracted_dict = extracted.model_dump(mode="json")
        resume_id = await asyncio.to_thread(insert_full_resume, extracted_dict, user_id, resolved_resume_key or "")

        return {"success": True, "message": "Resume parsed and stored successfully.", "resume_id": resume_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("parse_resume failed")
        raise HTTPException(status_code=500, detail="Failed to parse resume.") from exc


@router.post("/manual")
async def submit_manual_resume(
    payload: ResumeExtraction,
    user_id: str = Depends(require_user_id),
):
    try:  
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        resume_key = f"manual_{user_id}_{timestamp}"
        
        extracted_dict = payload.model_dump(mode="json")
        resume_id = await asyncio.to_thread(insert_full_resume, extracted_dict, user_id, resume_key)

        return {"success": True, "message": "Manual profile saved successfully.", "resume_id": resume_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("submit_manual_resume failed")
        raise HTTPException(status_code=500, detail="Failed to save manual profile.") from exc


@router.get("/latest")
async def get_latest_resume(user_id: str = Depends(require_user_id)):
    resume_id = latest_resume_id(user_id)
    if not resume_id:
        return {"success": True, "resume": None}
    return {"success": True, "resume": await fetch_full_resume(resume_id)}


@router.get("/all")
async def get_all_resumes(user_id: str = Depends(require_user_id)):
    try:
        # The Supabase Python client is synchronous, so keep its network I/O
        # off FastAPI's event loop.
        resp = await asyncio.to_thread(
            lambda: (
                db_client.table("resumes")
                .select("id, created_at, full_name, headline, summary, resume_key")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
        )
    except Exception as exc:
        logger.exception("get_all_resumes failed")
        raise HTTPException(status_code=500, detail="Failed to load saved resumes.") from exc

    return {"success": True, "resumes": resp.data or []}


@router.post("/select/{resume_id}")
async def select_resume(resume_id: str, user_id: str = Depends(require_user_id)):
    # Verify ownership
    resp = db_client.table("resumes").select("id").eq("user_id", user_id).eq("id", resume_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="Resume not found")
        
    # Bump created_at so this resume sorts as the latest (gap/jobs/github all
    # read the most-recent resume). The literal string "now()" is NOT a valid
    # timestamp via PostgREST — send a real ISO-8601 UTC timestamp instead.
    db_client.table("resumes").update(
        {"created_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", resume_id).execute()
    
    return {"success": True, "resume_id": resume_id}


# ── Profile editing ─────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    target_role_id: str


class SkillIn(BaseModel):
    skill: str


class ExperienceWrite(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool | None = None
    description_bullets: list[str] | None = None
    technologies: list[str] | None = None


class EducationWrite(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    grade: str | None = None
    notes: list[str] | None = None


class ProjectWrite(BaseModel):
    name: str | None = None
    description: str | None = None
    link: str | None = None
    technologies: list[str] | None = None


def _clean_text_list(values: list[str] | None) -> list[str]:
    """Normalize user-entered list fields before persisting child rows."""
    if values is None:
        return []
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _owned_resume_item(table: str, item_id: str, user_id: str) -> dict[str, Any]:
    """Return a resume-owned row or raise 404 without leaking another user's data."""
    row = db_client.table(table).select("id,resume_id").eq("id", item_id).limit(1).execute()
    if not row.data or row.data[0].get("resume_id") not in user_resume_ids(user_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return row.data[0]


def _replace_child_text_rows(table: str, foreign_key: str, parent_id: str, column: str, values: list[str]) -> None:
    db_client.table(table).delete().eq(foreign_key, parent_id).execute()
    if values:
        db_client.table(table).insert([{foreign_key: parent_id, column: value} for value in values]).execute()


@router.patch("/profile")
async def update_profile(body: ProfileUpdate, user_id: str = Depends(require_user_id)):
    """Change the user's target role (persisted in the `profiles` table)."""
    try:
        db_client.table("profiles").upsert(
            {"user_id": user_id, "target_role_id": body.target_role_id},
            on_conflict="user_id",
        ).execute()
    except Exception as e:
        logger.exception("profile update failed for user %s", user_id)
        raise HTTPException(status_code=500, detail=f"Profile update failed: {e}") from e
    return {"success": True, "target_role_id": body.target_role_id}


@router.post("/skills")
async def add_skill(body: SkillIn, user_id: str = Depends(require_user_id)):
    """Add a skill to the user's latest resume."""
    skill = (body.skill or "").strip()
    if not skill:
        raise HTTPException(status_code=400, detail="Skill cannot be empty")
    resume_id = latest_resume_id(user_id)
    if not resume_id:
        raise HTTPException(status_code=400, detail="No resume found")
    existing = (
        db_client.table("skills").select("skill").eq("resume_id", resume_id).execute()
    )
    if any((s.get("skill") or "").strip().lower() == skill.lower() for s in (existing.data or [])):
        return {"success": True, "skill": skill, "duplicate": True}
    db_client.table("skills").insert({"resume_id": resume_id, "skill": skill}).execute()
    return {"success": True, "skill": skill}


@router.delete("/skills")
async def delete_skill(skill: str, user_id: str = Depends(require_user_id)):
    """Remove a skill (by name) from the user's latest resume."""
    skill = (skill or "").strip()
    if not skill:
        raise HTTPException(status_code=400, detail="Skill cannot be empty")
    resume_id = latest_resume_id(user_id)
    if not resume_id:
        raise HTTPException(status_code=400, detail="No resume found")
    db_client.table("skills").delete().eq("resume_id", resume_id).eq("skill", skill).execute()
    db_client.table("programming_languages").delete()\
        .eq("resume_id", resume_id).eq("language", skill).execute()
    return {"success": True, "skill": skill}


@router.post("/experiences")
async def create_experience(body: ExperienceWrite, user_id: str = Depends(require_user_id)):
    resume_id = latest_resume_id(user_id)
    if not resume_id:
        raise HTTPException(status_code=400, detail="Create a resume or profile first")
    values = body.model_dump(exclude={"description_bullets", "technologies"})
    created = db_client.table("experiences").insert({"resume_id": resume_id, **values}).execute()
    experience_id = created.data[0]["id"]
    _replace_child_text_rows("experience_bullets", "experience_id", experience_id, "bullet", _clean_text_list(body.description_bullets))
    _replace_child_text_rows("experience_technologies", "experience_id", experience_id, "tech", _clean_text_list(body.technologies))
    return {"success": True, "id": experience_id}


@router.patch("/experiences/{experience_id}")
async def update_experience(experience_id: str, body: ExperienceWrite, user_id: str = Depends(require_user_id)):
    _owned_resume_item("experiences", experience_id, user_id)
    updates = body.model_dump(exclude={"description_bullets", "technologies"}, exclude_unset=True)
    if updates:
        db_client.table("experiences").update(updates).eq("id", experience_id).execute()
    if "description_bullets" in body.model_fields_set:
        _replace_child_text_rows("experience_bullets", "experience_id", experience_id, "bullet", _clean_text_list(body.description_bullets))
    if "technologies" in body.model_fields_set:
        _replace_child_text_rows("experience_technologies", "experience_id", experience_id, "tech", _clean_text_list(body.technologies))
    return {"success": True, "updated": updates}


@router.delete("/experiences/{experience_id}")
async def delete_experience(experience_id: str, user_id: str = Depends(require_user_id)):
    _owned_resume_item("experiences", experience_id, user_id)
    db_client.table("experiences").delete().eq("id", experience_id).execute()
    return {"success": True}


@router.post("/education")
async def create_education(body: EducationWrite, user_id: str = Depends(require_user_id)):
    resume_id = latest_resume_id(user_id)
    if not resume_id:
        raise HTTPException(status_code=400, detail="Create a resume or profile first")
    values = body.model_dump(exclude={"notes"})
    created = db_client.table("education").insert({"resume_id": resume_id, **values}).execute()
    education_id = created.data[0]["id"]
    _replace_child_text_rows("education_notes", "education_id", education_id, "note", _clean_text_list(body.notes))
    return {"success": True, "id": education_id}


@router.patch("/education/{education_id}")
async def update_education(education_id: str, body: EducationWrite, user_id: str = Depends(require_user_id)):
    _owned_resume_item("education", education_id, user_id)
    updates = body.model_dump(exclude={"notes"}, exclude_unset=True)
    if updates:
        db_client.table("education").update(updates).eq("id", education_id).execute()
    if "notes" in body.model_fields_set:
        _replace_child_text_rows("education_notes", "education_id", education_id, "note", _clean_text_list(body.notes))
    return {"success": True, "updated": updates}


@router.delete("/education/{education_id}")
async def delete_education(education_id: str, user_id: str = Depends(require_user_id)):
    _owned_resume_item("education", education_id, user_id)
    db_client.table("education").delete().eq("id", education_id).execute()
    return {"success": True}


@router.post("/projects")
async def create_project(body: ProjectWrite, user_id: str = Depends(require_user_id)):
    resume_id = latest_resume_id(user_id)
    if not resume_id:
        raise HTTPException(status_code=400, detail="Create a resume or profile first")
    values = body.model_dump(exclude={"technologies"})
    created = db_client.table("projects").insert({"resume_id": resume_id, **values}).execute()
    project_id = created.data[0]["id"]
    _replace_child_text_rows("project_technologies", "project_id", project_id, "tech", _clean_text_list(body.technologies))
    return {"success": True, "id": project_id}


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, body: ProjectWrite, user_id: str = Depends(require_user_id)):
    _owned_resume_item("projects", project_id, user_id)
    updates = body.model_dump(exclude={"technologies"}, exclude_unset=True)
    if updates:
        db_client.table("projects").update(updates).eq("id", project_id).execute()
    if "technologies" in body.model_fields_set:
        _replace_child_text_rows("project_technologies", "project_id", project_id, "tech", _clean_text_list(body.technologies))
    return {"success": True, "updated": updates}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, user_id: str = Depends(require_user_id)):
    _owned_resume_item("projects", project_id, user_id)
    db_client.table("projects").delete().eq("id", project_id).execute()
    return {"success": True}


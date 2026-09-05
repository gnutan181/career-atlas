import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

class GitHubOAuthCallback(BaseModel):
    code: str

class RepoSelection(BaseModel):
    repos: List[str] = Field(min_length=1, max_length=10)

    @field_validator("repos")
    @classmethod
    def validate_repo_names(cls, repos: List[str]) -> List[str]:
        if len(set(repos)) != len(repos):
            raise ValueError("Each repository may be selected only once.")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) for repo in repos):
            raise ValueError("Repository names must be in owner/repository format.")
        return repos

class SkillAction(BaseModel):
    evidence_ids: List[str] = Field(min_length=1, max_length=100)

class GitHubRepoInfo(BaseModel):
    name: str
    owner: str
    url: str
    description: Optional[str] = None
    stargazerCount: int
    pushedAt: Optional[str] = None
    primaryLanguage: Optional[str] = None
    isOwner: bool

class GitHubReposResponse(BaseModel):
    success: bool
    repos: List[GitHubRepoInfo]

class AnalysisResponse(BaseModel):
    success: bool
    summary: str
    coding_behavior: str
    skills: List[str]

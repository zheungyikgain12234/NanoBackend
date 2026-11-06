"""
byoa_adk/schema.py
~~~~~~~~~~~~~~~~~~
Pydantic v2 models that define the wire contract between the BYOA platform
and any agent built with this ADK.  Both sides import from here so the
contract is a single source of truth.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    """Terminal statuses an agent MUST use in TaskResult."""
    SUCCESS  = "success"
    FAILURE  = "failure"
    DEGRADED = "degraded"   # partial result; platform may retry or continue


class AgentStatus(str, Enum):
    """Agent-level liveness states returned by /health."""
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    UNHEALTHY = "unhealthy"


# ---------------------------------------------------------------------------
# Platform -> Agent: task call
# ---------------------------------------------------------------------------

class TaskPayload(BaseModel):
    """
    Sent by the platform orchestrator in the body of POST /run.

    Fields
    ------
    task_id     : Unique ID for this invocation (UUID4 string).
    goal_id     : Parent goal this task belongs to.
    capability  : The specific capability the platform is invoking.
                  Must appear in the agent's declared capability list.
    inputs      : Arbitrary dict of inputs — agent-specific content.
    timeout_secs: How long the platform will wait before marking the
                  agent DEGRADED.  ADK enforces this internally.
    hmac_sig    : HMAC-SHA256 hex digest of the canonical payload body
                  (everything except this field), signed with the shared
                  secret.  ADK verifies this before invoking run().
    timestamp   : Unix timestamp used in HMAC canonical message.
    """
    task_id      : str             = Field(..., description="UUID4")
    goal_id      : str             = Field(..., description="Parent goal UUID4")
    capability   : str             = Field(..., description="Capability being invoked")
    inputs       : Dict[str, Any]  = Field(default_factory=dict)
    timeout_secs : int             = Field(default=60, ge=1, le=3600)
    hmac_sig     : str             = Field(..., description="HMAC-SHA256 hex of canonical body")
    timestamp    : int             = Field(default=0, description="Unix timestamp for HMAC")

    @field_validator("capability")
    @classmethod
    def capability_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("capability must not be blank")
        return v.strip().lower()


# ---------------------------------------------------------------------------
# Agent -> Platform: task result
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    """
    Returned by the agent in the body of the POST /run response (HTTP 200).
    The platform rejects responses that do not conform to this schema.

    Fields
    ------
    task_id   : Must echo back the task_id from TaskPayload.
    status    : Terminal status of this task.
    outputs   : Arbitrary dict of results — agent-specific content.
    error_msg : Populated only when status != SUCCESS.
    duration_ms: Wall-clock time the agent spent on run(), in ms.
    """
    task_id     : str             = Field(...)
    status      : TaskStatus      = Field(...)
    outputs     : Dict[str, Any]  = Field(default_factory=dict)
    error_msg   : Optional[str]   = Field(default=None)
    duration_ms : Optional[int]   = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Agent manifest (Agent.md inner JSON block)
# ---------------------------------------------------------------------------

ALLOWED_CAPABILITIES: List[str] = [
    # Extend this allowlist as the platform grows.
    "log-ingestion", "log-analysis", "log-export",
    "web-crawl", "web-scrape", "web-search",
    "data-fetch", "data-transform", "data-export",
    "text-summarize", "text-classify", "text-extract",
    "image-analyse", "image-generate", "code-execute", "notify",
    "knowledge-retrieve", "combine", "data-matching",
]

class AgentManifest(BaseModel):
    """
    The structured JSON block inside Agent.md.
    Written by manifest.py, consumed by the platform registration server.

    Field names match exactly what the registration server validates:
        agent_id, display_name, version, endpoint_url,
        health_check_path, invoke_path, invoke_method,
        invoke_content_type, capabilities, input_schema, output_schema
    """
    # Identity
    agent_id         : str            = Field(..., pattern=r"^[a-z0-9\-]+$")
    display_name     : str            = Field(...)
    version          : str            = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description      : str            = Field(..., min_length=10)

    # Routing
    endpoint_url     : str            = Field(..., description="Base HTTPS URL")
    health_check_path: str            = Field(default="/health")
    invoke_path      : str            = Field(default="/run")
    invoke_method    : str            = Field(default="POST")
    invoke_content_type: str          = Field(default="application/json")

    # Capabilities + schemas
    capabilities     : List[str]      = Field(..., min_length=1)
    input_schema     : Dict[str, Any] = Field(default_factory=dict)
    output_schema    : Dict[str, Any] = Field(default_factory=lambda: {"type": "object"})

    # ADK metadata
    adk_version      : str            = Field(default="1.0.0")

    # LLM backend this agent declares it uses -- optional, since not every
    # agent calls an LLM at all (e.g. fx-agent is a plain REST call).
    # If provided, it must be one of the platform's approved providers.
    llm_base_url     : Optional[str]  = Field(default=None)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, caps: List[str]) -> List[str]:
        unknown = set(caps) - set(ALLOWED_CAPABILITIES)
        if unknown:
            raise ValueError(
                f"Unknown capabilities: {unknown}.\n"
                f"Allowed: {ALLOWED_CAPABILITIES}"
            )
        return caps

    @field_validator("llm_base_url")
    @classmethod
    def require_approved_llm_provider(cls, v: Optional[str]) -> Optional[str]:
        # An agent that doesn't call an LLM at all simply omits this field --
        # only validated when a value is actually declared.
        if v is None:
            return v
        from .llm_client import ALLOWED_LLM_BASE_URLS, _is_allowed
        if not _is_allowed(v):
            raise ValueError(
                f'llm_base_url {v!r} is not an approved model provider. '
                f'Allowed: {ALLOWED_LLM_BASE_URLS}'
            )
        return v

    @field_validator("endpoint_url")
    @classmethod
    def require_https(cls, v: str) -> str:
        # An agent's data is only as safe as the pipe it travels over.
        # Refusing plain http:// here means an insecure endpoint can never
        # even get as far as being registered, let alone dispatched to.
        if not v.startswith("https://"):
            raise ValueError(
                f'endpoint_url must use HTTPS (data must be encrypted in transit). Got: {v!r}'
            )
        return v

    @field_validator("invoke_method")
    @classmethod
    def validate_invoke_method(cls, v: str) -> str:
        if v.upper() not in {"POST", "GET"}:
            raise ValueError(f"invoke_method must be POST or GET, got: {v}")
        return v.upper()

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        for param, defn in v.items():
            if not isinstance(defn, dict):
                raise ValueError(f"input_schema.{param} must be an object")
            if "type" not in defn:
                raise ValueError(f"input_schema.{param} missing \"type\"")
            if "required" not in defn:
                raise ValueError(f"input_schema.{param} missing \"required\"")
        return v


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    name         : str
    version      : str
    status       : AgentStatus
    capabilities : List[str]
    adk_version  : str = "1.0.0"

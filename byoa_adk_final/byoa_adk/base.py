"""
byoa_adk/base.py
~~~~~~~~~~~~~~~~
Agent base class — the primary developer-facing surface of the ADK.
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import ClassVar, List, Optional

from .dag      import DAG
from .manifest import generate_agent_md
from .schema   import (
    AgentManifest, AgentStatus, HealthResponse,
    TaskPayload, TaskResult, TaskStatus,
)
from .security import HMACMiddleware, issue_agent_token, verify_request

logger = logging.getLogger("byoa_adk.agent")


class AgentConfigError(Exception):
    """Raised when required class attributes are missing or invalid."""


class Agent:
    """
    Base class for all BYOA-compatible agents.

    Required class attributes
    -------------------------
    name         : str        — kebab-case agent name, e.g. "my-agent"
    version      : str        — semver string, e.g. "1.0.0"
    description  : str        — at least 10 characters
    capabilities : List[str]  — subset of ALLOWED_CAPABILITIES from schema.py

    Optional class attributes
    -------------------------
    dag          : DAG | None — internal Kanban DAG (advisory for orchestrator)

    Environment variables consumed
    ------------------------------
    BYOA_SECRET   : Shared secret for HMAC signing (required at serve-time).
    AGENT_ENDPOINT: Public base URL for this agent (required for manifest).
    """

    # -- Developer sets these on the subclass --
    name        : ClassVar[str]
    version     : ClassVar[str]
    description : ClassVar[str]
    capabilities: ClassVar[List[str]]
    dag         : ClassVar[Optional[DAG]] = None

    # Which LLM backend this agent uses, if any -- omit entirely for
    # agents that don't call an LLM (e.g. a plain REST-calling agent).
    # Validated against the platform's approved provider allowlist by
    # AgentManifest itself (schema.py) the moment generate_manifest()
    # below actually builds one -- this is a single, plain declaration,
    # not something needing the property-shadowing pattern below, since
    # there's no separate call-time-argument path for it to reconcile with.
    llm_base_url: ClassVar[Optional[str]] = None

    # -- Populated by the FIRST call to generate_manifest(); reused by
    #    subsequent calls (e.g. the live /agent.md route) so the schema
    #    only needs to be declared once, in the developer's own setup code,
    #    not duplicated at every call site. --
    _cached_input_schema  : Optional[dict] = None
    _cached_output_schema : Optional[dict] = None
    _cached_endpoint      : Optional[str]  = None

    @property
    def output_schema(self) -> dict:
        """
        Exposes the schema last passed to generate_manifest(output_schema=...)
        so server.py's Response Contract Validator can actually see it.

        Before this property existed, `hasattr(agent, 'output_schema')` in
        server.py was always False -- Agent never defined that name, only
        `_cached_output_schema` -- so every /run response skipped contract
        validation entirely and returned as if it had passed. This property
        is what makes that hasattr() check True and the validator load-bearing.

        A subclass MAY instead declare `output_schema` directly as a plain
        class attribute (right next to `capabilities`) -- Python's normal
        attribute lookup means that plain value then shadows this property
        entirely for that subclass, so generate_manifest() and the Response
        Contract Validator both read the SAME single declaration, and it can
        never drift from what's actually enforced. See input_schema below
        for the same pattern on the request side.
        """
        return self._cached_output_schema or {}

    @property
    def input_schema(self) -> dict:
        """
        Exposes the schema last passed to generate_manifest(input_schema=...),
        read by server.py's Request Contract Validator before run() executes.

        Same shadowing pattern as output_schema: a subclass can instead
        declare `input_schema` directly as a plain class attribute, making
        that one declaration the single source of truth for both what
        Agent.md advertises AND what's actually enforced against real
        incoming requests -- the two can no longer silently disagree.
        """
        return self._cached_input_schema or {}

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        """Validate required class attributes at subclass definition time."""
        super().__init_subclass__(**kwargs)
        required = ("name", "version", "description", "capabilities")
        missing  = [r for r in required if not getattr(cls, r, None)]
        if missing:
            raise AgentConfigError(
                f"{cls.__name__} is missing required class attributes: {missing}"
            )
        # Validate the manifest at definition time — before platform contact.
        AgentManifest(
            agent_id     = cls.name,
            display_name = cls.name,
            version      = cls.version,
            description  = cls.description,
            capabilities = cls.capabilities,
            endpoint_url = "https://placeholder",
        )
        logger.debug("Agent class %s validated OK.", cls.__name__)

    # ---------------------------------------------------------------
    # Developer implements this
    # ---------------------------------------------------------------

    def run(self, payload: TaskPayload) -> TaskResult:
        """
        Override this method with your agent's core logic.

        MUST return a TaskResult.  Raising an exception here will be caught
        by _timed_run and returned as a FAILURE TaskResult — the agent will
        NOT crash the Flask server.
        """
        raise NotImplementedError("Agent.run() must be implemented by the subclass.")

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _validate_capability(self, capability: str) -> None:
        """Reject calls for capabilities the agent didn't declare."""
        if capability not in self.capabilities:
            raise ValueError(
                f"Agent {self.name!r} does not support capability {capability!r}. "
                f"Declared: {self.capabilities}"
            )

    def _timed_run(self, payload: TaskPayload) -> TaskResult:
        """
        Execute run() directly — no ThreadPoolExecutor.

        ThreadPoolExecutor deadlocks on Kaggle notebooks because the
        notebook kernel holds the GIL and never yields to the submitted
        thread. Direct invocation avoids this entirely.

        Timeout enforcement is delegated to the platform's AGENT_CALL_TIMEOUT
        and the agent's own request timeout to Ollama/external services.
        """
        t0 = time.monotonic()
        try:
            result = self.run(payload)
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            return result
        except Exception as exc:
            logger.exception("Unhandled exception in run(): %s", exc)
            return TaskResult(
                task_id     = payload.task_id,
                status      = TaskStatus.FAILURE,
                error_msg   = str(exc),
                duration_ms = int((time.monotonic() - t0) * 1000),
            )

    # ---------------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------------

    def generate_manifest(
        self,
        endpoint: Optional[str] = None,
        output_path: Optional[Path] = None,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
    ) -> str:
        """
        Generate Agent.md in the registration server format.

        Parameters
        ----------
        endpoint      : Public base HTTPS URL. Falls back to $AGENT_ENDPOINT,
                        then to the last endpoint passed to this method.
        output_path   : Write to this path (default: ./Agent.md).
        input_schema  : Dict of { param: { type, required } }.
                        e.g. {"text": {"type": "string", "required": True}}
                        If omitted, reuses the schema from the most recent
                        call that provided one (e.g. the developer's own
                        setup code), so routes like /agent.md that call
                        this with no arguments still serve the real schema
                        instead of silently falling back to {}.
        output_schema : Dict with at least {"type": "object"}.
                        Same fallback-to-cache behaviour as input_schema.
        """
        # Cache whichever schema was explicitly provided this call, so a
        # later no-argument call (e.g. from the /agent.md route) can reuse
        # it instead of falling back to an empty schema.
        from . import __version__  # deferred: package __init__ imports this module

        if input_schema is not None:
            self.__class__._cached_input_schema = input_schema
        if output_schema is not None:
            self.__class__._cached_output_schema = output_schema
        if endpoint is not None:
            self.__class__._cached_endpoint = endpoint

        resolved_input_schema  = input_schema  if input_schema  is not None else self.input_schema
        resolved_output_schema = output_schema if output_schema is not None else self.output_schema
        resolved_endpoint      = (
            endpoint
            or os.environ.get("AGENT_ENDPOINT")
            or self._cached_endpoint
            or "http://localhost:5000"
        )

        manifest = AgentManifest(
            agent_id          = self.name,
            display_name      = self.name,
            version           = self.version,
            description       = self.description,
            endpoint_url      = resolved_endpoint,
            health_check_path = "/health",
            invoke_path       = "/run",
            invoke_method     = "POST",
            invoke_content_type = "application/json",
            capabilities      = self.capabilities,
            input_schema      = resolved_input_schema or {},
            output_schema     = resolved_output_schema or {"type": "object"},
            adk_version       = __version__,
            llm_base_url      = self.llm_base_url,
        )
        dag_steps = self.dag.to_dict() if self.dag else None
        path      = output_path or Path("Agent.md")
        content   = generate_agent_md(manifest, dag_steps, output_path=path)
        logger.info("Agent.md written to %s", path)
        return content

    # ---------------------------------------------------------------
    # Flask server
    # ---------------------------------------------------------------

    def serve(self, port: int = 5000, host: str = "127.0.0.1") -> None:
        """
        Start the Flask server with HMAC middleware.
        Reads $BYOA_SECRET from the environment (required).
        """
        from .server import build_flask_app
        secret = os.environ.get("BYOA_SECRET")
        if not secret:
            raise RuntimeError(
                "$BYOA_SECRET is not set.  "
                "Set it to the shared secret provided by the platform."
            )
        app = build_flask_app(self, secret)
        logger.info("Starting %s v%s on %s:%d", self.name, self.version, host, port)
        app.run(host=host, port=port)

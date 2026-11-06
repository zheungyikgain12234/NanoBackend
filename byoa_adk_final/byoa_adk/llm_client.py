"""
byoa_adk/llm_client.py
~~~~~~~~~~~~~~~~~~~~~~~
The one sanctioned way for an agent to call out to an LLM backend.

IMPORTANT LIMIT, stated plainly: this is a code-level guardrail, not a
network-level guarantee. It works by being the easy, obvious path for
an agent's own run() to use -- it does NOT and CANNOT stop code in that
same process from bypassing it entirely (e.g. calling `requests.post`
directly to any URL). Nothing running inside a process you don't fully
control can be forced to use this.

For agents you host yourself, this is genuinely effective: you review
the code, and this is what a compliant agent is written to use. For a
third party's own agent, running on their own server, this is only as
strong as their willingness to actually use it -- same limit as every
other "what happens once it's on their machine" question already
covered for this platform. Real, unbypassable enforcement for that case
requires a NETWORK-level control (an egress firewall or forward proxy
restricting which hosts the process can reach at all) -- outside what
any Python library can do from inside the process being restricted.
"""
from __future__ import annotations
import os
from typing import Optional

import requests as _requests

# The only LLM backends this platform currently sanctions. Read from
# the environment where it makes sense to (VLLM_BASE_URL already exists
# and varies by deployment); the external providers are fixed constants.
ALLOWED_LLM_BASE_URLS = [
    os.environ.get("VLLM_BASE_URL", "http://localhost:11500"),
    "https://api.openai.com",
    "https://api.anthropic.com",
]


class UnapprovedLLMEndpointError(ValueError):
    """Raised when code tries to call an LLM base URL not on the allowlist."""


def _is_allowed(base_url: str) -> bool:
    base_url = base_url.rstrip("/")
    for allowed in ALLOWED_LLM_BASE_URLS:
        allowed = allowed.rstrip("/")
        if base_url == allowed or base_url.startswith(allowed + "/"):
            return True
    return False


def call_llm(
    base_url: str,
    path: str,
    json: dict,
    headers: Optional[dict] = None,
    timeout: int = 60,
):
    """
    The sanctioned entry point for any outbound LLM call. Raises
    UnapprovedLLMEndpointError immediately -- BEFORE any network
    connection is even attempted -- if base_url isn't on the allowlist.

    Usage (inside an agent's run()):
        resp = call_llm(VLLM_BASE_URL, "/v1/chat/completions", json={...})
    """
    if not _is_allowed(base_url):
        raise UnapprovedLLMEndpointError(
            f"{base_url!r} is not an approved LLM endpoint. "
            f"Allowed: {ALLOWED_LLM_BASE_URLS}"
        )
    url = base_url.rstrip("/") + path
    return _requests.post(url, json=json, headers=headers or {}, timeout=timeout)

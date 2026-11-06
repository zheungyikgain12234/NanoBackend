"""
byoa_adk/manifest.py
~~~~~~~~~~~~~~~~~~~~
Generate and parse Agent.md manifest files.

The Agent.md format produced here matches exactly what the BYOA
registration server expects — field names are the registration server's
field names, not ADK-internal names.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

from .schema import AgentManifest

# Regex to extract the first ```json ... ``` fenced block from Agent.md
_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def generate_agent_md(
    manifest: AgentManifest,
    dag_steps: Optional[list] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Render an Agent.md string from a validated AgentManifest.
    The JSON block produced is directly submittable to POST /register.

    Parameters
    ----------
    manifest    : Validated AgentManifest instance.
    dag_steps   : Optional list from DAG.to_dict() — informational only.
    output_path : If provided, write Agent.md here (default: ./Agent.md).
    """
    data = manifest.model_dump()

    # Include DAG as informational metadata if provided
    if dag_steps:
        data["dag"] = dag_steps

    json_block = json.dumps(data, indent=2)

    content = (
        f"# Agent: {manifest.agent_id}\n\n"
        f"{manifest.description}\n\n"
        f"```json\n{json_block}\n```\n"
    )

    path = output_path or Path("Agent.md")
    Path(path).write_text(content)

    return content


def parse_agent_md(text: str) -> AgentManifest:
    """
    Parse an Agent.md string and return a validated AgentManifest.
    Raises ValueError if no JSON block found.
    Raises pydantic.ValidationError if the manifest fields are invalid.
    """
    match = _FENCE_RE.search(text)
    if not match:
        raise ValueError("No ```json ... ``` block found in Agent.md.")
    data = json.loads(match.group(1))
    return AgentManifest.model_validate(data)

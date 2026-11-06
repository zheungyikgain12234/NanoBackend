"""
byoa_adk/cli.py
~~~~~~~~~~~~~~~
byoa-adk command-line tool.
Install via pip, then run:  byoa-adk --help
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import click
import requests

from .manifest import parse_agent_md
from .schema   import TaskPayload


@click.group()
def cli():
    """BYOA Agent Development Kit CLI."""


# ---------------------------------------------------------------------------
# byoa-adk init
# ---------------------------------------------------------------------------

AGENT_TEMPLATE = """\
from byoa_adk import Agent, DAG, Step
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus

class {class_name}(Agent):
    name         = "{agent_name}"
    version      = "1.0.0"
    description  = "Replace this with a description of at least 10 characters."
    capabilities = ["data-fetch"]  # pick from schema.ALLOWED_CAPABILITIES

    # Optional: declare an internal Kanban DAG
    dag = DAG([
        Step("fetch",   depends_on=[]),
        Step("process", depends_on=["fetch"]),
    ])

    def run(self, payload: TaskPayload) -> TaskResult:
        # TODO: implement your agent logic here
        return TaskResult(
            task_id = payload.task_id,
            status  = TaskStatus.SUCCESS,
            outputs = {{"message": "Hello from {agent_name}!"}},
        )

if __name__ == "__main__":
    {class_name}().serve(port=5000)
"""

@cli.command()
@click.argument("name")
def init(name: str):
    """Scaffold a new agent project directory."""
    out = Path(name)
    if out.exists():
        click.echo(f"Directory {name!r} already exists.", err=True)
        sys.exit(1)
    out.mkdir()
    class_name = "".join(p.title() for p in name.split("-")) + "Agent"
    (out / "agent.py").write_text(
        AGENT_TEMPLATE.format(class_name=class_name, agent_name=name)
    )
    (out / "requirements.txt").write_text("byoa-adk>=1.0.0\n")
    (out / ".env.example").write_text(
        "BYOA_SECRET=<shared-secret-from-platform>\n"
        "AGENT_ENDPOINT=https://<your-public-url>\n"
    )
    click.echo(f"Created agent project in ./{name}/")
    click.echo(f"  Edit {name}/agent.py, then run: byoa-adk validate")


# ---------------------------------------------------------------------------
# byoa-adk validate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", default="Agent.md", type=click.Path(exists=True))
def validate(path: str):
    """Validate Agent.md without contacting the platform."""
    text = Path(path).read_text()
    try:
        manifest = parse_agent_md(text)
        click.echo(f"OK  {manifest.agent_id} v{manifest.version}")
        click.echo(f"    Capabilities: {manifest.capabilities}")
        click.echo(f"    Endpoint:     {manifest.endpoint_url}")
    except Exception as exc:
        click.echo(f"FAIL  {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# byoa-adk register
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("platform_url")
@click.argument("agent_md", default="Agent.md", type=click.Path(exists=True))
def register(platform_url: str, agent_md: str):
    """POST Agent.md to the platform's /register endpoint."""
    import os
    from .security import issue_agent_token

    text = Path(agent_md).read_text()
    try:
        manifest = parse_agent_md(text)
    except Exception as exc:
        click.echo(f"Local validation failed: {exc}", err=True)
        sys.exit(1)

    secret = os.environ.get("BYOA_SECRET")
    if not secret:
        click.echo("$BYOA_SECRET is not set -- cannot prove agent identity to the registration server.", err=True)
        sys.exit(1)
    token = issue_agent_token(secret, manifest.agent_id)

    url = platform_url.rstrip("/") + "/register"
    click.echo(f"Registering {manifest.agent_id} at {url} ...")
    try:
        resp = requests.post(
            url,
            files={"agent_md": ("Agent.md", text, "text/markdown")},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.ok:
            click.echo(f"Registered OK: {resp.json()}")
        else:
            click.echo(f"Registration failed ({resp.status_code}): {resp.text}", err=True)
            sys.exit(1)
    except requests.RequestException as exc:
        click.echo(f"Network error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# byoa-adk simulate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("payload_file", type=click.Path(exists=True))
@click.argument("agent_module", default="agent")
def simulate(payload_file: str, agent_module: str):
    """
    Dry-run run() with a JSON payload file.  No HTTP, no HMAC check.

    Example payload.json:
      {"task_id":"t1","goal_id":"g1","capability":"data-fetch",
       "inputs":{"url":"http://example.com"},"timeout_secs":30,"hmac_sig":"skip"}
    """
    import importlib
    sys.path.insert(0, ".")
    mod = importlib.import_module(agent_module)
    from byoa_adk.base import Agent as _Base
    agent_cls = next(
        (v for v in vars(mod).values()
         if isinstance(v, type) and issubclass(v, _Base) and v is not _Base),
        None,
    )
    if not agent_cls:
        click.echo("No Agent subclass found in module.", err=True)
        sys.exit(1)

    raw = json.loads(Path(payload_file).read_text())
    raw.setdefault("hmac_sig", "simulate-bypass")
    payload = TaskPayload.model_validate(raw)

    agent   = agent_cls()
    result  = agent._timed_run(payload)
    click.echo(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    cli()

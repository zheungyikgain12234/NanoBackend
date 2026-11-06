# BYOA Agent Development Kit (byoa_adk)

The ADK used to build, secure, and serve agents on the BYOA platform.

## Install

```bash
pip install -e .
```

For Postgres-backed audit logging (optional -- file-based logging works
without this):

```bash
pip install -e ".[postgres-audit]"
```

## What's included

- `Agent` -- base class for building an agent: input/output validation,
  crash containment, manifest generation.
- `DAG` / `Step` -- declare and actually execute an agent's internal steps
  in dependency order.
- HMAC request signing/verification, with replay protection.
- Structured audit logging (Postgres or JSONL file), including automatic
  detection of suspicious request patterns (e.g. a burst of failed
  signature attempts).
- `build_flask_app()` -- wires an `Agent` into a real Flask app with
  `/health`, `/agent.md`, and `/run`, all security enforced automatically.

## CLI

Installing this package puts a `byoa-adk` command on your PATH:

```bash
byoa-adk init my-new-agent      # scaffold a new agent project
byoa-adk simulate payload.json agent.py   # test run() with no HTTP/HMAC
byoa-adk validate Agent.md      # validate a manifest locally
byoa-adk register <url> Agent.md  # register with a platform (requires $BYOA_SECRET)
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

#!/usr/bin/env bash
# Start the agent: loads .env, activates the venv, serves on $AGENT_PORT.
cd "$(dirname "$0")" && . venv/bin/activate && set -a && . ./.env && set +a
exec python ocr_agent.py

"""
BYOA Offensive-Text Classifier Agent -- standalone service.

Classifies input text as offensive or not offensive, using the
platform's own vLLM instance.

Deliberately built as a reference example of doing this correctly:
  - input_schema/output_schema/llm_base_url all declared ONCE, directly
    on the class -- the single source of truth generate_manifest() reads
    from, and the SAME thing the ADK enforces against real requests.
  - Uses call_llm() (the sanctioned allowlist-checked path from
    byoa_adk.llm_client), not a raw requests.post() call -- unlike
    summarize_agent.py, this agent's runtime LLM calls are genuinely
    covered by the platform's approved-provider enforcement.
"""
import json
import os
import re
import sys

from dotenv import load_dotenv
from waitress import serve

from byoa_adk import Agent
from byoa_adk.llm_client import call_llm
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

load_dotenv()

PORT = int(os.environ.get("PORT", "5500"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
os.environ["BYOA_SECRET"] = BYOA_SECRET
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://offensive-text.zackloh.dev")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-fp8")

_GUIDED_SCHEMA = {
    "type": "object",
    "properties": {
        "is_offensive": {"type": "boolean"},
        "confidence":   {"type": "number"},
        "reason":       {"type": "string"},
    },
    "required": ["is_offensive", "confidence", "reason"],
}

_PROMPT_TEMPLATE = """Classify the following text as offensive or not offensive. \
Consider hate speech, harassment, slurs, and threats as offensive. Ordinary \
disagreement, criticism, or strong-but-civil language is NOT offensive.

Text to classify:
\"\"\"
{text}
\"\"\"

Respond with ONLY a single valid JSON object, and nothing else -- no markdown \
code fences, no preamble, no explanation outside the JSON. Example of the \
exact format required:
{{"is_offensive": false, "confidence": 0.95, "reason": "Brief explanation here."}}"""


def _parse_model_response(content: str) -> dict:
    """
    Parse the model's response into {"is_offensive", "confidence", "reason"}.

    Tries, in order: straight JSON, JSON inside markdown fences, a {...}
    block found anywhere in the text, and finally plain "key: value" lines
    -- this last format is a REAL fallback this model has been observed to
    produce when structured-output enforcement (guided_json) isn't applied
    by the serving backend, despite being requested.
    """
    content = content.strip()

    fenced = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
    for candidate in (content, fenced):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    result = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, _, value = line.partition(':')
        key, value = key.strip(), value.strip()
        if value.lower() in ('true', 'false'):
            result[key] = value.lower() == 'true'
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    if {'is_offensive', 'confidence', 'reason'} <= result.keys():
        return result

    raise ValueError(f'Could not parse model response as structured data: {content!r}')


class OffensiveTextAgent(Agent):
    """
    Classifies input text as offensive or not offensive.
    Inputs : {"text": "some string to classify"}
    Outputs: {"is_offensive": bool, "confidence": float, "reason": str}
    """
    name         = "offensive-text-agent"
    version      = "1.0.0"
    description  = "Classifies input text as offensive or not offensive using an LLM."
    capabilities = ["text-classify"]

    # Declared once, here -- generate_manifest() reads this directly, and
    # the ADK enforces this SAME declaration against every real request
    # before run() ever executes.
    input_schema = {
        "text": {"type": "string", "required": True},
    }
    output_schema = {
        "type": "object",
        "properties": {
            "is_offensive": {"type": "boolean"},
            "confidence":   {"type": "number"},
            "reason":       {"type": "string"},
        },
    }

    # Declared honestly, matching what run() actually calls -- checked
    # against the platform's approved-provider allowlist the moment
    # Agent.md is generated.
    llm_base_url = VLLM_BASE_URL

    def run(self, payload: TaskPayload) -> TaskResult:
        text = payload.inputs["text"]  # already guaranteed present by input_schema enforcement

        try:
            resp = call_llm(
                VLLM_BASE_URL,
                "/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": _PROMPT_TEMPLATE.format(text=text)}],
                    "temperature": 0,
                    "guided_json": _GUIDED_SCHEMA,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = _parse_model_response(content)
        except Exception as exc:
            return TaskResult(
                task_id   = payload.task_id,
                status    = TaskStatus.FAILURE,
                error_msg = f"Classification failed: {exc}",
            )

        return TaskResult(
            task_id = payload.task_id,
            status  = TaskStatus.SUCCESS,
            outputs = {
                "is_offensive": bool(result["is_offensive"]),
                "confidence":   float(result["confidence"]),
                "reason":       str(result["reason"]),
            },
        )


agent = OffensiveTextAgent()
agent.generate_manifest(endpoint=AGENT_ENDPOINT)
app = build_flask_app(agent, BYOA_SECRET)


def _selftest():
    """Run with: python offensive_text_agent.py --selftest"""
    from byoa_adk.security import sign_request

    def _payload(task_id, text, timeout=30):
        sig, ts = sign_request(BYOA_SECRET, task_id, "goal_test", "text-classify")
        return TaskPayload(task_id=task_id, goal_id="goal_test",
                           capability="text-classify",
                           inputs={"text": text},
                           timeout_secs=timeout, hmac_sig=sig, timestamp=ts)

    cases = [
        "You did a great job on this project, thank you!",
        "I disagree with your approach, but let's discuss it.",
    ]
    for i, text in enumerate(cases, 1):
        r = agent._timed_run(_payload(f"t{i}", text))
        assert r.status == TaskStatus.SUCCESS, f"FAIL: {r.error_msg}"
        o = r.outputs
        print(f'PASS [{i}] offensive={o["is_offensive"]} confidence={o["confidence"]:.2f} -- {o["reason"]}')

    # Confirm input contract enforcement rejects a missing 'text' field
    from byoa_adk.security import sign_request
    sig, ts = sign_request(BYOA_SECRET, "t_bad", "goal_test", "text-classify")
    bad_payload = TaskPayload(task_id="t_bad", goal_id="goal_test", capability="text-classify",
                              inputs={}, timeout_secs=30, hmac_sig=sig, timestamp=ts)
    # Note: input contract enforcement runs in server.py's /run route, not
    # in _timed_run() directly -- this call bypasses it, same as _selftest
    # bypassing HMAC. Exercise the real endpoint (not this script) to see
    # the 422 rejection in practice.

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)

    print("=" * 60)
    print(f"Offensive Text Agent -- {agent.name} v{agent.version}")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT}")
    print(f"  GET  {AGENT_ENDPOINT}/health")
    print(f"  GET  {AGENT_ENDPOINT}/agent.md")
    print(f"  POST {AGENT_ENDPOINT}/run")
    print("=" * 60)
    serve(app, host="0.0.0.0", port=PORT, threads=4)

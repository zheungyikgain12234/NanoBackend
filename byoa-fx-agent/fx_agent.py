"""
BYOA FX Agent -- standalone GX10 service.

Same FxAgent logic as the Kaggle notebook (adk-fx-agent.ipynb, Cell 3),
unchanged. What's different from the Kaggle version:
  - No pyngrok code in this process. Public exposure is handled by the
    systemd-managed `cloudflared` unit already running on this box, which
    proxies https://fx.zackloh.dev -> http://localhost:PORT.
  - No `while True: sleep(60)` keepalive cell -- systemd keeps this alive instead.
  - No local smoke-test cell -- run `python fx_agent.py --selftest` for that,
    or just hit the systemd-managed endpoint directly once it's up.
  - Config comes from a .env file instead of hardcoded notebook cell values.
"""
import os
import sys

import requests as _req
from dotenv import load_dotenv
from waitress import serve

from byoa_adk import Agent, DAG, Step
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

load_dotenv()

PORT = int(os.environ.get("PORT", "5403"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
os.environ["BYOA_SECRET"] = BYOA_SECRET
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://fx.zackloh.dev")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500")

class FxAgent(Agent):
    """
    Live currency conversion agent.
    Inputs : {"from": "USD", "to": "SGD", "amount": 100}
    Outputs: {"from", "to", "amount", "rate", "converted"}
    Uses open.er-api.com -- no API key required.
    """
    name         = "fx-agent"
    version      = "1.0.0"
    description  = "Fetches live FX rates and converts currency amounts."
    capabilities = ["data-fetch", "data-transform"]
    llm_base_url = VLLM_BASE_URL
    
    dag = DAG([
        Step("fetch",   depends_on=[],        description="Fetch rate from open.er-api.com"),
        Step("convert", depends_on=["fetch"], description="Apply conversion formula"),
    ])

    def run(self, payload: TaskPayload) -> TaskResult:
        # Each function below corresponds 1:1 to a Step declared in self.dag
        # above, by name. self.dag.run() -- not this method -- decides what
        # order they execute in, based on each Step's depends_on. If a Step
        # is ever added, removed, or reordered in the DAG, execution follows
        # automatically; nothing here needs to change to match it.
        context = {"payload": payload}

        def fetch(ctx):
            p = ctx["payload"]
            frm = p.inputs.get("from", "USD").upper()
            try:
                resp = _req.get(
                    f"https://open.er-api.com/v6/latest/{frm}",
                    timeout=5,
                )
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"Rate fetch failed: {exc}") from exc
            return {"frm": frm, "rates": resp.json().get("rates", {})}

        def convert(ctx):
            p = ctx["payload"]
            to     = p.inputs.get("to", "SGD").upper()
            amount = float(p.inputs.get("amount", 1))
            rates  = ctx["rates"]
            if to not in rates:
                raise RuntimeError(f"Unknown currency: {to}")
            rate = rates[to]
            return {
                "to"        : to,
                "amount"    : amount,
                "rate"      : rate,
                "converted" : round(amount * rate, 4),
            }

        try:
            context = self.dag.run(context, {"fetch": fetch, "convert": convert})
        except Exception as exc:
            return TaskResult(
                task_id   = payload.task_id,
                status    = TaskStatus.FAILURE,
                error_msg = str(exc),
            )

        return TaskResult(
            task_id = payload.task_id,
            status  = TaskStatus.SUCCESS,
            outputs = {
                "from"      : context["frm"],
                "to"        : context["to"],
                "amount"    : context["amount"],
                "rate"      : context["rate"],
                "converted" : context["converted"],
            },
        )


agent = FxAgent()

agent.generate_manifest(
    endpoint      = AGENT_ENDPOINT,
    input_schema  = {
        "from":   {"type": "string", "required": True},
        "to":     {"type": "string", "required": True},
        "amount": {"type": "number", "required": True},
    },
    output_schema = {
        "type": "object",
        "properties": {
            "from":      {"type": "string"},
            "to":        {"type": "string"},
            "amount":    {"type": "number"},
            "rate":      {"type": "number"},
            "converted": {"type": "number"},
        }
    },
)

app = build_flask_app(agent, BYOA_SECRET)


def _selftest():
    """Equivalent of notebook Cell 4, runnable on the box: python fx_agent.py --selftest"""
    from byoa_adk.security import sign_request

    def _payload(task_id, frm, to, amount, timeout=10):
        sig, ts = sign_request(BYOA_SECRET, task_id, "goal_test", "data-fetch")
        return TaskPayload(task_id=task_id, goal_id="goal_test",
                           capability="data-fetch",
                           inputs={"from": frm, "to": to, "amount": amount},
                           timeout_secs=timeout, hmac_sig=sig, timestamp=ts)

    cases = [
        ("USD", "SGD", 250),
        ("EUR", "JPY", 100),
        ("GBP", "MYR", 500),
    ]

    for i, (frm, to, amount) in enumerate(cases, 1):
        r = agent._timed_run(_payload(f"t{i}", frm, to, amount))
        assert r.status == TaskStatus.SUCCESS, f"FAIL: {r.error_msg}"
        o = r.outputs
        print(f'PASS [{i}] {o["amount"]} {o["from"]} -> {o["converted"]} {o["to"]}  (rate={o["rate"]})')

    r_bad = agent._timed_run(_payload("t_bad", "USD", "XXX", 10))
    assert r_bad.status == TaskStatus.FAILURE
    print(f"PASS unknown currency rejected: {r_bad.error_msg}")

    try:
        agent._validate_capability("web-crawl")
    except ValueError:
        print("PASS wrong capability rejected")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)

    print("=" * 60)
    print(f"FX Agent -- {agent.name} v{agent.version}")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT} (cloudflared, managed by systemd)")
    print(f"  GET  {AGENT_ENDPOINT}/health")
    print(f"  GET  {AGENT_ENDPOINT}/agent.md")
    print(f"  POST {AGENT_ENDPOINT}/run")
    print("=" * 60)
    serve(app, host="0.0.0.0", port=PORT, threads=4)

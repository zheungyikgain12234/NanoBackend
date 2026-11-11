import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import code_generate as cg  # noqa: E402

PLAIN = '''"""demo -- a BYOA agent."""
import os
import sys

from byoa_adk import Agent, attach_async_routes
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

PORT = int(os.environ.get("PORT", "5000"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "x")


class DemoAgent(Agent):
    """Counts words in text."""
    name         = "demo"
    version      = "1.0.0"
    description  = "Counts the words in a text."
    capabilities = ["text-extract"]
    input_schema = {
        "text": {"type": "string", "required": True},
    }
    output_schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
        },
    }

    def run(self, payload: TaskPayload) -> TaskResult:
        # payload.inputs is already checked against input_schema.
        text = payload.inputs["text"]

        try:
            # TODO: replace this with your agent's real logic.
            outputs = {
                "count": 0,
            }
        except Exception as exc:
            return TaskResult(task_id=payload.task_id, status=TaskStatus.FAILURE, error_msg=str(exc))

        return TaskResult(task_id=payload.task_id, status=TaskStatus.SUCCESS, outputs=outputs)


agent = DemoAgent()
agent.generate_manifest(endpoint="https://demo.example.com")
app = build_flask_app(agent, BYOA_SECRET, extra_protected_paths={"/run_async"})
attach_async_routes(app, agent)
'''

DAG_SRC = '''import os
from byoa_adk import Agent, DAG, Step
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app


class DemoAgent(Agent):
    """Two steps."""
    name         = "demo"
    version      = "1.0.0"
    description  = "Cleans then counts text."
    capabilities = ["text-extract"]
    input_schema = {"text": {"type": "string", "required": True}}
    output_schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    dag = DAG([
        Step("clean", depends_on=[]),
        Step("count", depends_on=["clean"]),
    ])

    def run(self, payload: TaskPayload) -> TaskResult:
        context = {"payload": payload}

        def step_clean(ctx):
            # Strip the text.
            # TODO: do this step's work; read earlier results from ctx.
            return {}

        def step_count(ctx):
            # TODO: do this step's work; read earlier results from ctx.
            return {}

        try:
            context = self.dag.run(context, {"clean": step_clean, "count": step_count})
        except Exception as exc:
            return TaskResult(task_id=payload.task_id, status=TaskStatus.FAILURE, error_msg=str(exc))

        return TaskResult(task_id=payload.task_id, status=TaskStatus.SUCCESS,
                          outputs={"count": context.get("count", 0)})


agent = DemoAgent()
app = build_flask_app(agent, "s")
'''

GOOD_PLAIN = """### IMPORTS
import re
### REGION run_body
words = re.findall(r"\\w+", text)
outputs = {"count": len(words)}
"""


def fake(*replies):
    it = iter(replies)
    calls = []
    def llm(messages):
        calls.append(list(messages))
        return next(it)
    llm.calls = calls
    return llm


class T(unittest.TestCase):
    def test_regions_plain_and_dag(self):
        self.assertEqual([r["id"] for r in cg.find_regions(PLAIN)], ["run_body"])
        self.assertEqual([r["id"] for r in cg.find_regions(DAG_SRC)], ["step_clean", "step_count"])

    def test_only_regions_and_imports_change(self):
        out = cg.generate("count words", PLAIN, llm=fake(GOOD_PLAIN))
        self.assertTrue(out["check"]["ok"], out["check"])
        new, old = out["code"].split("\n"), PLAIN.split("\n")
        added = [l for l in new if l not in old]
        self.assertEqual(sorted(added), sorted(["import re", '            words = re.findall(r"\\w+", text)',
                                                '            outputs = {"count": len(words)}']))
        # every ORIGINAL line outside the TODO try-body survives, in order
        start = old.index("        try:")
        end = old.index("        except Exception as exc:")
        outside = old[:start + 1] + old[end:]
        it = iter(new)
        self.assertTrue(all(any(l == n for n in it) for l in outside))
        self.assertEqual(out["added_imports"], ["import re"])

    def test_dag_steps_filled(self):
        reply = ("### REGION step_clean\nreturn {'text': ctx['payload'].inputs['text'].strip()}\n"
                 "### REGION step_count\nreturn {'count': len(ctx['text'].split())}\n")
        out = cg.generate("x", DAG_SRC, llm=fake(reply))
        self.assertIn("ctx['text'].split()", out["code"])
        self.assertIn('context = self.dag.run(context, {"clean": step_clean', out["code"])

    def test_retry_on_syntax_error_then_ok(self):
        llm = fake("### REGION run_body\noutputs = {", GOOD_PLAIN)
        out = cg.generate("x", PLAIN, llm=llm)
        self.assertEqual(out["attempts"], 2)
        self.assertIn("syntax error", llm.calls[1][-1]["content"])

    def test_forbidden_things_are_rejected(self):
        bad_call = "### REGION run_body\noutputs = {'count': eval('1')}"
        bad_import = "### IMPORTS\nimport subprocess\n### REGION run_body\noutputs = {'count': 1}"
        for bad in (bad_call, bad_import):
            llm = fake(bad, bad, bad)
            with self.assertRaises(cg.GenerateError):
                cg.generate("x", PLAIN, llm=llm)

    def test_missing_outputs_and_region_rejected(self):
        with self.assertRaises(cg.GenerateError):
            cg.generate("x", PLAIN, llm=fake(*["### REGION run_body\ny = 1"] * 3))
        with self.assertRaises(cg.GenerateError):
            cg.generate("x", PLAIN, llm=fake(*["### IMPORTS\n"] * 3))

    def test_import_in_region_and_bad_import_line(self):
        with self.assertRaises(cg.GenerateError):
            cg.generate("x", PLAIN, llm=fake(*["### REGION run_body\nimport re\noutputs = {}"] * 3))
        with self.assertRaises(ValueError):
            cg._validate_imports(["import os; os.system('x')"])
        with self.assertRaises(ValueError):
            cg._validate_imports(["from os import *"])

    def test_code_with_nothing_to_fill(self):
        with self.assertRaises(cg.GenerateError):
            cg.find_regions("class A(Agent):\n    def run(self, payload):\n        return 1\n")
        with self.assertRaises(cg.GenerateError):
            cg.find_regions("def (:")

    def test_echoed_end_marker_is_dropped(self):
        reply = GOOD_PLAIN + "### END REGION run_body\n"
        out = cg.generate("x", PLAIN, llm=fake(reply))
        self.assertNotIn("END REGION", out["code"])
        out = cg.generate("x", PLAIN, llm=fake(GOOD_PLAIN + "# <<< END REGION run_body >>>\n"))
        self.assertNotIn("<<<", out["code"])

    def test_bare_hash_line_is_dropped(self):
        out = cg.generate("x", PLAIN, llm=fake(GOOD_PLAIN + "###\n"))
        self.assertNotIn("###", out["code"])
        # a real, short comment is kept
        reply = GOOD_PLAIN.replace("words =", "# split into words\nwords =")
        self.assertIn("# split into words", cg.generate("x", PLAIN, llm=fake(reply))["code"])

    def test_indented_body_and_trailing_repeated_header(self):
        reply = ('### IMPORTS\n### REGION run_body\n            words = text.split()\n'
                 '            outputs = {"count": len(words)}\n### REGION run_body')
        out = cg.generate("x", PLAIN, llm=fake(reply))
        self.assertTrue(out["check"]["ok"])
        self.assertIn('            words = text.split()', out["code"])
        self.assertEqual(out["attempts"], 1)

    def test_fenced_reply_is_accepted(self):
        out = cg.generate("x", PLAIN, llm=fake("```\n" + GOOD_PLAIN + "```"))
        self.assertTrue(out["check"]["ok"])

    def test_job_flow_and_owner_isolation(self):
        import time
        orig = cg._call_llm
        cg._call_llm = fake(GOOD_PLAIN)
        try:
            jid = cg.start_job("alice", "count words", PLAIN)
            self.assertIsNone(cg.get_job("bob", jid))
            for _ in range(100):
                j = cg.get_job("alice", jid)
                if j["status"] in ("done", "error"):
                    break
                time.sleep(0.05)
            self.assertEqual(j["status"], "done", j)
            self.assertIn("outputs", j["result"]["code"])
        finally:
            cg._call_llm = orig
        with self.assertRaises(cg.GenerateError):
            cg.start_job("alice", "  ", PLAIN)


if __name__ == "__main__":
    unittest.main()

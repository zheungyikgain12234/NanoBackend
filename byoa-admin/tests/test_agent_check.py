import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_check import check_agent_code  # noqa: E402

GOOD = '''
from byoa_adk import Agent, DAG, Step, attach_async_routes
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

class MyAgent(Agent):
    name = "my-agent"
    version = "1.0.0"
    description = "Does something useful."
    capabilities = ["text-summarize"]
    input_schema = {"text": {"type": "string", "required": True}}
    output_schema = {"type": "object", "properties": {"r": {"type": "string"}}}
    dag = DAG([Step("a"), Step("b", depends_on=["a"])])

    def run(self, payload: TaskPayload) -> TaskResult:
        ctx = self.dag.run({}, {"a": lambda c: {}, "b": lambda c: {}})
        return TaskResult(task_id=payload.task_id, status=TaskStatus.SUCCESS, outputs={"r": ""})

agent = MyAgent()
app = build_flask_app(agent, "s", extra_protected_paths={"/run_async"})
attach_async_routes(app, agent)
'''


def codes(src, sev=None):
    r = check_agent_code(src)
    return [i["code"] for i in r["issues"] if sev is None or i["severity"] == sev], r["ok"]


class T(unittest.TestCase):
    def test_good_agent_is_clean(self):
        c, ok = codes(GOOD)
        self.assertEqual((c, ok), ([], True))

    def test_syntax_error_has_line(self):
        r = check_agent_code("class A(Agent:\n  pass\n")
        self.assertFalse(r["ok"])
        self.assertEqual(r["issues"][0]["code"], "syntax")

    def test_no_agent_class_and_missing_attrs(self):
        self.assertIn("no-agent-class", codes("x = 1")[0])
        c, ok = codes("class A(Agent):\n    name='a-b'\n")
        self.assertFalse(ok)
        for m in ("missing-version", "missing-description", "missing-capabilities", "no-run"):
            self.assertIn(m, c)

    def test_bad_values(self):
        bad = GOOD.replace('"my-agent"', '"My Agent"').replace('"1.0.0"', '"1.0"').replace(
            '"Does something useful."', '"short"').replace('["text-summarize"]', '["teleport"]')
        c, ok = codes(bad, "error")
        self.assertFalse(ok)
        for m in ("bad-name", "bad-version", "bad-description", "unknown-capability"):
            self.assertIn(m, c)

    def test_dag_cycle_unknown_dep_duplicate(self):
        cyc = GOOD.replace('Step("a"), Step("b", depends_on=["a"])', 'Step("a", depends_on=["b"]), Step("b", depends_on=["a"])')
        self.assertIn("dag-cycle", codes(cyc)[0])
        unk = GOOD.replace('depends_on=["a"]', 'depends_on=["zzz"]')
        self.assertIn("dag-unknown-dep", codes(unk)[0])
        dup = GOOD.replace('Step("b", depends_on=["a"])', 'Step("a")')
        self.assertIn("dag-duplicate", codes(dup)[0])

    def test_async_without_protection_is_an_error(self):
        bad = GOOD.replace(', extra_protected_paths={"/run_async"}', "")
        c, ok = codes(bad, "error")
        self.assertIn("async-unprotected", c)
        self.assertFalse(ok)

    def test_async_with_unused_capability_warns(self):
        self.assertIn("async-not-used", codes(GOOD.replace("text-summarize", "web-search"))[0])

    def test_default_secret_and_llm_warnings(self):
        src = GOOD + '\nimport os\nS = os.environ.get("BYOA_SECRET", "hunter2")\ncall_llm("u", "/p", json={})\n'
        c, _ = codes(src, "warning")
        self.assertIn("default-secret", c)
        self.assertIn("llm-undeclared", c)

    def test_dag_declared_but_not_run_warns(self):
        self.assertIn("dag-unused", codes(GOOD.replace("self.dag.run({}, {\"a\": lambda c: {}, \"b\": lambda c: {}})", "None"))[0])

    def test_schema_passed_to_generate_manifest_is_not_flagged(self):
        src = GOOD.replace('    input_schema = {"text": {"type": "string", "required": True}}\n', "") + \
            '\nagent.generate_manifest(endpoint="https://x", input_schema={})\n'
        self.assertNotIn("no-input_schema", codes(src)[0])

    def test_code_is_never_executed(self):
        marker = "/tmp/claude-agent-check-should-not-exist"
        if os.path.exists(marker):
            os.remove(marker)
        check_agent_code(GOOD + f'\nopen({marker!r}, "w").write("ran")\nraise SystemExit(1)\n')
        self.assertFalse(os.path.exists(marker))

    def test_oversize_and_deep_nesting_do_not_crash(self):
        self.assertFalse(check_agent_code("x=1\n" * 100000)["ok"])
        r = check_agent_code("x = " + "(" * 5000 + "1" + ")" * 5000)
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

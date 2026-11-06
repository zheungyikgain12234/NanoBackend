"""Runs a REAL byoa_adk agent on a local port and probes it with verify-secret,
so the 401 = wrong secret / 403 = right secret assumption is tested for real."""
import os, sys, tempfile, threading, unittest
from unittest import mock

os.environ.update(PLATFORM_DB_URL="postgresql://x/x", ADMIN_JWT_SECRET="s" * 40,
                  ADMIN_AUDIT_LOG=os.path.join(tempfile.mkdtemp(), "audit.log"),
                  AUDIT_LOG_PATH=os.path.join(tempfile.mkdtemp(), "agent_audit.log"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import admin_api as a  # noqa: E402
from byoa_adk import Agent, TaskResult, TaskStatus, build_flask_app  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

RAN = []


class Fx(Agent):
    name = "fx-agent"; version = "1.0.0"; description = "Fetches FX rates."; capabilities = ["data-fetch"]
    def run(self, p):
        RAN.append(p); return TaskResult(task_id=p.task_id, status=TaskStatus.SUCCESS)


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = make_server("127.0.0.1", 0, build_flask_app(Fx(), "the-agents-real-secret"))
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_port

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    MD = ('# Agent: fx-agent\n\nx\n\n```json\n{"agent_id":"fx-agent","display_name":"fx-agent","version":"1.0.0",'
          '"description":"Fetches FX rates.","endpoint_url":"https://agent.test","capabilities":["data-fetch"]}\n```\n')

    def login(self, c):
        with mock.patch.object(a, "verify_login", return_value=True):
            return {"Authorization": "Bearer " + c.post("/auth/login", json={"username": "u", "password": "p"}).json["token"]}

    def probe(self, stored_secret, port=None):
        """Manifests must be https, so route the fake https host to the local agent."""
        real_post = a.requests.post
        local = port or self.port
        def route(url, **kw):
            return real_post(url.replace("https://agent.test", "http://127.0.0.1:%d" % local), **kw)
        c = a.app.test_client()
        with mock.patch.object(a, "get_agent_secret", return_value=stored_secret), \
             mock.patch.object(a.requests, "post", side_effect=route):
            return c.post("/verify-secret", json={"agent_md": self.MD}, headers=self.login(c)).json

    def test_right_secret_is_ok_and_no_agent_code_runs(self):
        self.assertEqual(self.probe("the-agents-real-secret")["status"], "ok")
        self.assertEqual(RAN, [])

    def test_wrong_secret_is_mismatch(self):
        self.assertEqual(self.probe("some-other-secret")["status"], "mismatch")

    def test_agent_down_is_unreachable(self):
        self.assertEqual(self.probe("x", port=9)["status"], "unreachable")

    def test_not_onboarded(self):
        c = a.app.test_client()
        with mock.patch.object(a, "get_agent_secret", return_value=None):
            r = c.post("/verify-secret", json={"agent_md": self.MD}, headers=self.login(c))
        self.assertEqual(r.status_code, 409)

    def test_non_https_endpoint_rejected(self):
        c = a.app.test_client()
        r = c.post("/verify-secret", json={"agent_md": self.MD.replace("https://", "http://")}, headers=self.login(c))
        self.assertEqual(r.status_code, 422)

    def test_requires_login(self):
        self.assertEqual(a.app.test_client().post("/verify-secret", json={}).status_code, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)

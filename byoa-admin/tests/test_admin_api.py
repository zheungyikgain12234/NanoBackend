import os, sys, tempfile, unittest
from unittest import mock

os.environ.update(PLATFORM_DB_URL="postgresql://x/x", ADMIN_JWT_SECRET="s" * 40,
                  ADMIN_AUDIT_LOG=os.path.join(tempfile.mkdtemp(), "audit.log"),
                  ADMIN_CORS_ORIGINS="http://localhost:8080")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import admin_api as a  # noqa: E402

MD = '# Agent: fx-agent\n\nx\n\n```json\n{"agent_id":"fx-agent","display_name":"fx-agent","version":"1.0.0","description":"Fetches FX rates.","endpoint_url":"https://fx7.example.com","capabilities":["data-fetch"]}\n```\n'


class T(unittest.TestCase):
    def setUp(self):
        a._fails.clear()
        self.c = a.app.test_client()

    def login(self, ok=True):
        with mock.patch.object(a, "verify_login", return_value=ok):
            return self.c.post("/auth/login", json={"username": "zack", "password": "pw"})

    def auth(self):
        return {"Authorization": "Bearer " + self.login().json["token"]}

    def test_login_ok_and_bad(self):
        self.assertEqual(self.login().status_code, 200)
        self.assertEqual(self.login(False).status_code, 401)

    def test_lockout_after_5_failures(self):
        for _ in range(5):
            self.login(False)
        self.assertEqual(self.login(True).status_code, 429)

    def test_routes_need_session(self):
        for m, p in (("get", "/credentials"), ("post", "/onboard"), ("post", "/register"), ("get", "/agents"),
                     ("post", "/agents/fx-agent/revoke")):
            self.assertEqual(getattr(self.c, m)(p).status_code, 401, p)
        self.assertEqual(self.c.get("/credentials", headers={"Authorization": "Bearer junk"}).status_code, 401)

    def test_agent_token_not_accepted_as_session(self):
        from byoa_adk.security import issue_agent_token
        tok = issue_agent_token("s" * 40, "fx-agent")
        self.assertEqual(self.c.get("/credentials", headers={"Authorization": "Bearer " + tok}).status_code, 401)

    def test_onboard_returns_secret_once_and_validates(self):
        h = self.auth()
        self.assertEqual(self.c.post("/onboard", json={"agent_id": "Bad_ID"}, headers=h).status_code, 422)
        with mock.patch.object(a, "onboard_agent", return_value="SECRET123"):
            r = self.c.post("/onboard", json={"agent_id": "fx-agent", "owner": "internal"}, headers=h)
        self.assertEqual((r.status_code, r.json["secret"]), (201, "SECRET123"))
        with mock.patch.object(a, "onboard_agent", side_effect=ValueError("already")):
            self.assertEqual(self.c.post("/onboard", json={"agent_id": "fx-agent"}, headers=h).status_code, 409)

    def test_credentials_list_has_no_secret(self):
        row = [{"agent_id": "fx-agent", "owner": "i", "revoked": False, "created_at": None, "revoked_at": None}]
        with mock.patch.object(a, "list_credentials", return_value=row):
            r = self.c.get("/credentials", headers=self.auth())
        self.assertNotIn("secret", r.get_data(as_text=True))

    def test_register_signs_with_stored_secret(self):
        h = self.auth()
        up = mock.Mock(ok=True, status_code=200); up.json.return_value = {"status": "REGISTERED_ACTIVE"}
        with mock.patch.object(a, "get_agent_secret", return_value="agentsecret"), \
             mock.patch.object(a.requests, "post", return_value=up) as post:
            r = self.c.post("/register", json={"agent_md": MD}, headers=h)
        self.assertEqual((r.status_code, r.json["status"]), (200, "REGISTERED_ACTIVE"))
        import jwt
        tok = post.call_args.kwargs["headers"]["Authorization"][7:]
        self.assertEqual(jwt.decode(tok, "agentsecret", algorithms=["HS256"])["sub"], "fx-agent")
        self.assertNotIn("agentsecret", r.get_data(as_text=True))

    def test_register_not_onboarded_and_bad_md(self):
        h = self.auth()
        with mock.patch.object(a, "get_agent_secret", return_value=None):
            self.assertEqual(self.c.post("/register", json={"agent_md": MD}, headers=h).status_code, 409)
        self.assertEqual(self.c.post("/register", json={"agent_md": "nope"}, headers=h).status_code, 422)
        self.assertEqual(self.c.post("/register", json={}, headers=h).status_code, 400)

    def test_revoke(self):
        h = self.auth()
        with mock.patch.object(a, "revoke_agent", return_value=True):
            self.assertEqual(self.c.post("/agents/fx-agent/revoke", headers=h).status_code, 200)
        with mock.patch.object(a, "revoke_agent", return_value=False):
            self.assertEqual(self.c.post("/agents/fx-agent/revoke", headers=h).status_code, 404)

    def test_unregister_and_delete(self):
        h = self.auth()
        for m, p in (("post", "/agents/fx-agent/unregister"), ("delete", "/agents/fx-agent")):
            self.assertEqual(getattr(self.c, m)(p).status_code, 401, p)
        with mock.patch.object(a, "unregister_agent", return_value=2):
            self.assertEqual(self.c.post("/agents/fx-agent/unregister", headers=h).json["rows"], 2)
        with mock.patch.object(a, "unregister_agent", return_value=0):
            self.assertEqual(self.c.post("/agents/fx-agent/unregister", headers=h).status_code, 404)
        with mock.patch.object(a, "delete_agent", return_value={"registry_rows": 1, "credential_rows": 1}) as d:
            self.assertEqual(self.c.delete("/agents/fx-agent", json={}, headers=h).status_code, 400)
            self.assertEqual(self.c.delete("/agents/fx-agent", json={"confirm_agent_id": "other"}, headers=h).status_code, 400)
            d.assert_not_called()
            r = self.c.delete("/agents/fx-agent", json={"confirm_agent_id": "fx-agent"}, headers=h)
            self.assertEqual((r.status_code, r.json["deleted"]), (200, True))
        with mock.patch.object(a, "delete_agent", return_value={"registry_rows": 0, "credential_rows": 0}):
            self.assertEqual(self.c.delete("/agents/fx-agent", json={"confirm_agent_id": "fx-agent"}, headers=h).status_code, 404)

    def test_code_endpoints_need_login_and_work(self):
        self.assertEqual(self.c.get("/code/reference").status_code, 401)
        self.assertEqual(self.c.post("/code/check", json={"code": "x"}).status_code, 401)
        h = self.auth()
        ref = self.c.get("/code/reference", headers=h).json
        self.assertIn("data-fetch", ref["capabilities"])
        self.assertIn("text-summarize", ref["async_capabilities"])
        self.assertEqual(self.c.post("/code/check", json={}, headers=h).status_code, 400)
        r = self.c.post("/code/check", json={"code": "class A(Agent:"}, headers=h).json
        self.assertFalse(r["ok"])

    def test_cors_only_for_listed_origin(self):
        ok = self.c.options("/credentials", headers={"Origin": "http://localhost:8080", "Access-Control-Request-Method": "GET"})
        bad = self.c.options("/credentials", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
        self.assertEqual(ok.headers.get("Access-Control-Allow-Origin"), "http://localhost:8080")
        self.assertIsNone(bad.headers.get("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

# BYOA Admin API

Backend for the Flutter admin app in `../byoa-admin-ui`.

## Run (local only)
```bash
cd /opt/v6/byoa-admin
cp .env.example .env     # fill in PLATFORM_DB_URL and ADMIN_JWT_SECRET
./venv/bin/python create_admin.py <username>   # hidden password prompt, min 12 chars
./venv/bin/python admin_api.py                 # listens on 127.0.0.1:4150
./venv/bin/python tests/test_admin_api.py      # unit tests (mocked, no DB needed)
```
Admin users are created only with `create_admin.py`; there is no HTTP endpoint for it.
Do not expose this service publicly without HTTPS in front of it.

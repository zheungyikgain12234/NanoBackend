# nanowaka-fin-ocr-agent

BYOA/ADK agent: PDF(s) -> Markdown using the Hugging Face model
`zheungyikgain/nanowaka.fin.ocr` (finetune of ATH-MaaS/OvisOCR2). Pattern follows `/opt/v6/byoa-match`.

## Run
    ./run.sh          # loads .env, loads the model (~1 min), serves on :5008
Config lives in `.env` (see `.env.example`). Private HF repo? `hf auth login` or set `HF_TOKEN`.

## Endpoints
| Route | Auth | Use |
|---|---|---|
| `POST /ocr` | `X-Api-Key: $BYOA_SECRET`, or none when `OCR_REQUIRE_KEY=0` | multipart, field `file` (repeat for many PDFs). **Async by default** -> job_id; `?async=false` blocks |
| `GET /ocr_status/<job_id>` | `X-Api-Key` | poll the upload job (status running/done/error) |
| `POST /run`, `POST /run_async` | HMAC (ADK) | what the agent platform calls. JSON inputs: `files` [{filename, content_base64}], `paths`, `directory`, `dpi`, `max_new_tokens`, `include_markdown` |
| `GET /status/<job_id>` | none | poll `/run_async` |
| `GET /health`, `/agent.md`, `/model_status` | none | liveness, manifest, model loaded? |

Why two upload styles: the ADK's HMAC middleware only parses JSON bodies, so a raw file upload cannot
go through `/run`. `/ocr` is the Postman/curl-friendly route; `/run` is the one the platform will use.

Server-side `paths`/`directory` must be inside `OCR_ALLOWED_INPUT_ROOTS` (default `~/Documents:~/Downloads`).
Markdown is also saved to `output/<task_id>/`. Speed: ~30 s per dense page; use async for long PDFs.

## Test in Postman
Import `nanowaka-ocr-agent.postman_collection.json`, set collection variable `secret` to `BYOA_SECRET` from `.env`.
Request 2 returns a job_id (saved automatically); send request 3 until status is `done`. Request 4 = blocking version.

## Before registering on the platform
- Set `AGENT_ENDPOINT` in `.env` to the real HTTPS URL, restart (regenerates `Agent.md`).
- Capability is `text-extract` (from the ADK allowlist); change `BYOA_SECRET` to the platform's secret.
- `/ocr` is a convenience route outside the platform contract; expose it only on a trusted network, or drop it.

## Open /ocr for a browser frontend
`.env`: `OCR_REQUIRE_KEY=0` (no key on /ocr and /ocr_status) and `OCR_CORS_ORIGINS=*` (or a comma-separated list of origins).
This exposes the GPU to anyone who can reach the URL; /run and /run_async stay HMAC-protected.

## Response shape (`/ocr`, `/ocr_status`)
```json
{ "status": "done", "error": null, "model": "...",
  "markdown": "<all Markdown, once>",
  "summary": {"files": 1, "ok": 1, "flagged": 0, "errors": 0, "pages": 1},
  "results": [{"filename": "a.pdf", "status": "ok", "pages": 1, "flags": [], "seconds": 22.7}] }
```
Several PDFs: `markdown` joins them, each under `<!-- file: name -->`. Add `?detail=true` to also get each file's own
`results[].markdown`. `/run` / `/run_async` do the same (input `per_file_markdown: true`) and still return server paths.

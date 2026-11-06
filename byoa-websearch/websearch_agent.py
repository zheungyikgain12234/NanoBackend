"""
BYOA Web Search Agent -- standalone GX10 service.

Same WebSearchAgent logic as the Kaggle notebook (adk-websearch-agent.ipynb,
Cell 3/6), unchanged. What's different from the Kaggle version:
  - No pyngrok code in this process. Public exposure is handled by the
    systemd-managed `cloudflared` unit on this box, proxying
    https://websearch.zackloh.dev -> http://localhost:PORT.
  - No `while True: sleep(60)` keepalive cell -- systemd keeps this alive.
  - Config comes from a .env file instead of hardcoded notebook cell values.
"""
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as _req
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from waitress import serve

from byoa_adk import Agent, DAG, Step
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

load_dotenv()

PORT = int(os.environ.get("PORT", "5404"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
os.environ["BYOA_SECRET"] = BYOA_SECRET
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://websearch.zackloh.dev")

MIN_TEXT_LENGTH = int(os.environ.get("MIN_TEXT_LENGTH", "200"))
DEFAULT_MIN_RESULTS = int(os.environ.get("DEFAULT_MIN_RESULTS", "15"))
FETCH_TIMEOUT_SECS = int(os.environ.get("FETCH_TIMEOUT_SECS", "8"))
USER_AGENT = "byoa-websearch-agent/1.0 (+https://byoa.platform)"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _clean_text(soup: BeautifulSoup) -> str:
    """Strip boilerplate tags and return collapsed plain text."""
    for tag in soup(["script", "style", "noscript", "header",
                     "footer", "nav", "aside", "form", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _fetch_and_verify(url: str, min_length: int, timeout: int) -> dict:
    """
    Fetch a URL and check it has meaningful text.

    Returns dict:
        ok          : bool -- True if text meets min_length
        url         : str
        title       : str
        text_preview: str (first 300 chars of clean text)
        text_length : int
        error       : str | None
    """
    try:
        resp = _req.get(
            url,
            headers={"User-Agent": USER_AGENT,
                     "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout,
            allow_redirects=True,
        )
    except _req.exceptions.Timeout:
        return {"ok": False, "url": url, "title": "", "text_preview": "",
                "text_length": 0, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "url": url, "title": "", "text_preview": "",
                "text_length": 0, "error": str(e)[:120]}

    if resp.status_code >= 400:
        return {"ok": False, "url": url, "title": "", "text_preview": "",
                "text_length": 0, "error": f"HTTP {resp.status_code}"}

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type and "xml" not in content_type:
        return {"ok": False, "url": url, "title": "", "text_preview": "",
                "text_length": 0, "error": f"Non-HTML: {content_type[:60]}"}

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = _clean_text(soup)
    except Exception as e:
        return {"ok": False, "url": url, "title": "", "text_preview": "",
                "text_length": 0, "error": f"Parse error: {e}"}

    if len(text) < min_length:
        return {"ok": False, "url": url, "title": title, "text_preview": "",
                "text_length": len(text),
                "error": f"Text too short: {len(text)} chars (min {min_length})"}

    return {
        "ok": True,
        "url": url,
        "title": title,
        "text_preview": text[:300],
        "text_length": len(text),
        "error": None,
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class WebSearchAgent(Agent):
    """
    Searches DuckDuckGo for a query, fetches each candidate URL,
    verifies it contains real meaningful text (>= MIN_TEXT_LENGTH chars),
    and returns at least min_results verified URLs.

    Uses ThreadPoolExecutor to fetch multiple URLs concurrently -- keeps
    the run fast even when some URLs are slow or blocked.
    """
    name = "byoa-websearch-agent"
    version = "1.0.0"
    description = ("Searches DuckDuckGo for a query, verifies each result URL "
                   "has real meaningful content, and returns at least 15 URLs "
                   "with confirmed text. No API key required.")
    capabilities = ["web-search"]

    dag = DAG([
        Step("search", depends_on=[], description="Query DuckDuckGo for candidates"),
        Step("verify", depends_on=["search"], description="Fetch each URL and check content"),
        Step("collect", depends_on=["verify"], description="Assemble verified results list"),
    ])

    def run(self, payload: TaskPayload) -> TaskResult:
        query = str(payload.inputs.get("query", "")).strip()
        min_results = max(DEFAULT_MIN_RESULTS, int(payload.inputs.get("min_results", DEFAULT_MIN_RESULTS)))
        max_attempts = int(payload.inputs.get("max_attempts", 50))

        if not query:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg="inputs.query is required")

        min_results = max(1, min(min_results, 50))
        max_attempts = max(min_results, min(max_attempts, 100))

        # ── Step 1: search DuckDuckGo ────────────────────────────────────────
        fetch_count = min(max_attempts, max(min_results * 3, 30))
        candidates = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=fetch_count):
                    url = r.get("href", "")
                    title = r.get("title", "")
                    snippet = r.get("body", "")
                    if _is_valid_url(url):
                        candidates.append({
                            "url": url,
                            "title": title,
                            "snippet": snippet,
                        })
        except Exception as e:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"DuckDuckGo search failed: {e}")

        if not candidates:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"No search results returned for: {query!r}")

        # ── Step 2: verify URLs concurrently ────────────────────────────────
        verified = []
        failed_urls = []
        attempted = 0
        ddg_snippets = {c["url"]: c for c in candidates}

        to_check = candidates[:max_attempts]

        # Concurrency raised from 4 -> 10: this work is purely I/O-bound
        # (waiting on network responses), so more concurrent fetches means
        # faster wall-clock completion of the same verification checks --
        # no change to which URLs get checked, how they're verified, or the
        # early-exit-at-min_results logic below.
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(
                    _fetch_and_verify,
                    c["url"],
                    MIN_TEXT_LENGTH,
                    FETCH_TIMEOUT_SECS,
                ): c["url"]
                for c in to_check
            }

            for future in as_completed(futures):
                attempted += 1
                result = future.result()
                url = result["url"]
                original = ddg_snippets.get(url, {})

                if result["ok"]:
                    verified.append({
                        "url": url,
                        "title": result["title"] or original.get("title", ""),
                        "snippet": original.get("snippet", ""),
                        "text_preview": result["text_preview"],
                        "text_length": result["text_length"],
                    })
                else:
                    failed_urls.append({
                        "url": url,
                        "error": result["error"],
                    })

                if len(verified) >= min_results:
                    for f in futures:
                        f.cancel()
                    break

        # ── Step 3: check we met the minimum ────────────────────────────────
        if len(verified) < min_results:
            status = TaskStatus.DEGRADED
            error_msg = (
                f"Only found {len(verified)} verified URLs out of "
                f"{min_results} requested. "
                f"Checked {attempted} candidates from DuckDuckGo. "
                f"Many may be bot-protected or have thin content."
            )
        else:
            status = TaskStatus.SUCCESS
            error_msg = None

        return TaskResult(
            task_id=payload.task_id,
            status=status,
            error_msg=error_msg,
            outputs={
                "query": query,
                "results": verified[:min_results],
                "total_found": len(verified),
                "total_attempted": attempted,
                "failed_urls": failed_urls[:10],
            },
        )


agent = WebSearchAgent()

agent.generate_manifest(
    endpoint=AGENT_ENDPOINT,
    input_schema={
        "query": {"type": "string", "required": True},
        "min_results": {"type": "integer", "required": False},
        "max_attempts": {"type": "integer", "required": False},
    },
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "results": {"type": "array"},
            "total_found": {"type": "integer"},
            "total_attempted": {"type": "integer"},
            "failed_urls": {"type": "array"},
        },
    },
)

app = build_flask_app(agent, BYOA_SECRET)

if __name__ == "__main__":
    print("=" * 60)
    print(f"Web Search Agent -- {agent.name} v{agent.version}")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT} (cloudflared, managed by systemd)")
    print(f"  GET  {AGENT_ENDPOINT}/health")
    print(f"  GET  {AGENT_ENDPOINT}/agent.md")
    print(f"  POST {AGENT_ENDPOINT}/run")
    print("=" * 60)
    serve(app, host="0.0.0.0", port=PORT, threads=4)

"""
BYOA Web Crawl Agent -- standalone GX10 service.

Same WebCrawlAgent logic as the Kaggle notebook (adk-webcrawl-agent.ipynb, Cell 3),
unchanged. What's different from the Kaggle version:
  - No pyngrok / cloudflared code in this process. Public exposure is handled by
    the systemd-managed `cloudflared` unit already running on this box, which
    proxies https://webcrawl.zackloh.dev -> http://localhost:PORT.
  - No `while True: sleep(60)` keepalive cell -- systemd keeps this alive instead.
  - Config comes from a .env file instead of hardcoded notebook cell values.
"""
import os
from urllib.parse import urljoin, urlparse

import requests as _req
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from waitress import serve

from byoa_adk import Agent, DAG, Step
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

load_dotenv()

PORT = int(os.environ.get("PORT", "5402"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
os.environ["BYOA_SECRET"] = BYOA_SECRET
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://webcrawl.zackloh.dev")

USER_AGENT = "byoa-webcrawl-agent/1.0 (+https://byoa.platform)"
REQUEST_TIMEOUT_FALLBACK = 15
MAX_TEXT_CHARS = 20000


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _extract_links(soup: BeautifulSoup, base_url: str, limit: int) -> list:
    seen = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        if absolute not in seen and absolute.startswith(("http://", "https://")):
            seen.add(absolute)
            links.append(absolute)
        if len(links) >= limit:
            break
    return links


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    return text[:MAX_TEXT_CHARS]


class WebCrawlAgent(Agent):
    """
    Fetches a single URL and returns cleaned page text plus discovered links.
    Does not recursively crawl -- one page per task, by design, so the
    platform orchestrator controls crawl depth/fan-out at the DAG level.
    """
    name = "byoa-webcrawl-agent"
    version = "1.0.0"
    description = ("Fetches a URL, strips boilerplate HTML, and returns clean "
                   "page text plus discovered links. Stateless, single-page "
                   "fetch per task; no JS rendering.")
    capabilities = ["web-crawl"]

    dag = DAG([
        Step("validate", depends_on=[], description="Validate inputs.url"),
        Step("fetch", depends_on=["validate"], description="HTTP GET the target URL"),
        Step("parse", depends_on=["fetch"], description="Strip boilerplate, extract text + links"),
    ])

    def run(self, payload: TaskPayload) -> TaskResult:
        url = str(payload.inputs.get("url", "")).strip()
        max_links = payload.inputs.get("max_links", 20)

        if not url:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg="inputs.url is required")
        if not _is_valid_url(url):
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"inputs.url is not a valid http(s) URL: {url!r}")

        try:
            max_links = int(max_links)
        except (TypeError, ValueError):
            max_links = 20
        max_links = max(0, min(max_links, 100))

        timeout = max(1, int(payload.timeout_secs * 0.9)) if payload.timeout_secs else REQUEST_TIMEOUT_FALLBACK

        try:
            resp = _req.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
                allow_redirects=True,
            )
        except _req.exceptions.Timeout:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"Request timed out after {timeout}s")
        except _req.exceptions.ConnectionError as e:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"Connection failed: {e}")
        except Exception as e:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"Fetch failed: {e}")

        if resp.status_code >= 400:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"HTTP {resp.status_code} fetching {url}")

        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type and "xml" not in content_type:
            return TaskResult(
                task_id=payload.task_id,
                status=TaskStatus.DEGRADED,
                outputs={
                    "url": url,
                    "status_code": resp.status_code,
                    "warning": f"Non-HTML content-type: {content_type or 'unknown'}",
                },
            )

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            text = _clean_text(soup)
            links = _extract_links(soup, url, max_links)
        except Exception as e:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"Parsing failed: {e}")

        return TaskResult(
            task_id=payload.task_id,
            status=TaskStatus.SUCCESS,
            outputs={
                "url": url,
                "title": title,
                "text": text,
                "text_length": len(text),
                "links": links,
                "status_code": resp.status_code,
            },
        )


agent = WebCrawlAgent()

agent.generate_manifest(
    endpoint=AGENT_ENDPOINT,
    input_schema={
        "url": {"type": "string", "required": True},
        "max_links": {"type": "integer", "required": False},
    },
    output_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "text": {"type": "string"},
            "text_length": {"type": "integer"},
            "links": {"type": "array"},
            "status_code": {"type": "integer"},
        },
    },
)

app = build_flask_app(agent, BYOA_SECRET)

if __name__ == "__main__":
    print("=" * 60)
    print(f"Web Crawl Agent -- {agent.name} v{agent.version}")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT} (cloudflared, managed by systemd)")
    print(f"  GET  {AGENT_ENDPOINT}/health")
    print(f"  GET  {AGENT_ENDPOINT}/agent.md")
    print(f"  POST {AGENT_ENDPOINT}/run")
    print("=" * 60)
    serve(app, host="0.0.0.0", port=PORT, threads=4)

# Agent: byoa-webcrawl-agent

Fetches a URL, strips boilerplate HTML, and returns clean page text plus discovered links. Stateless, single-page fetch per task; no JS rendering.

```json
{
  "agent_id": "byoa-webcrawl-agent",
  "display_name": "byoa-webcrawl-agent",
  "version": "1.0.0",
  "description": "Fetches a URL, strips boilerplate HTML, and returns clean page text plus discovered links. Stateless, single-page fetch per task; no JS rendering.",
  "endpoint_url": "https://webcrawl7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "web-crawl"
  ],
  "input_schema": {
    "url": {
      "type": "string",
      "required": true
    },
    "max_links": {
      "type": "integer",
      "required": false
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string"
      },
      "title": {
        "type": "string"
      },
      "text": {
        "type": "string"
      },
      "text_length": {
        "type": "integer"
      },
      "links": {
        "type": "array"
      },
      "status_code": {
        "type": "integer"
      }
    }
  },
  "adk_version": "1.0.0",
  "llm_base_url": null,
  "dag": [
    {
      "name": "validate",
      "depends_on": [],
      "description": "Validate inputs.url",
      "capability": null
    },
    {
      "name": "fetch",
      "depends_on": [
        "validate"
      ],
      "description": "HTTP GET the target URL",
      "capability": null
    },
    {
      "name": "parse",
      "depends_on": [
        "fetch"
      ],
      "description": "Strip boilerplate, extract text + links",
      "capability": null
    }
  ]
}
```

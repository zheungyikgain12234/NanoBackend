# Agent: byoa-websearch-agent

Searches DuckDuckGo for a query, verifies each result URL has real meaningful content, and returns at least 15 URLs with confirmed text. No API key required.

```json
{
  "agent_id": "byoa-websearch-agent",
  "display_name": "byoa-websearch-agent",
  "version": "1.0.0",
  "description": "Searches DuckDuckGo for a query, verifies each result URL has real meaningful content, and returns at least 15 URLs with confirmed text. No API key required.",
  "endpoint_url": "https://websearch7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "web-search"
  ],
  "input_schema": {
    "query": {
      "type": "string",
      "required": true
    },
    "min_results": {
      "type": "integer",
      "required": false
    },
    "max_attempts": {
      "type": "integer",
      "required": false
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string"
      },
      "results": {
        "type": "array"
      },
      "total_found": {
        "type": "integer"
      },
      "total_attempted": {
        "type": "integer"
      },
      "failed_urls": {
        "type": "array"
      }
    }
  },
  "adk_version": "1.0.0",
  "llm_base_url": null,
  "dag": [
    {
      "name": "search",
      "depends_on": [],
      "description": "Query DuckDuckGo for candidates",
      "capability": null
    },
    {
      "name": "verify",
      "depends_on": [
        "search"
      ],
      "description": "Fetch each URL and check content",
      "capability": null
    },
    {
      "name": "collect",
      "depends_on": [
        "verify"
      ],
      "description": "Assemble verified results list",
      "capability": null
    }
  ]
}
```

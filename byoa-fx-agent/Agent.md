# Agent: fx-agent

Fetches live FX rates and converts currency amounts.

```json
{
  "agent_id": "fx-agent",
  "display_name": "fx-agent",
  "version": "1.0.0",
  "description": "Fetches live FX rates and converts currency amounts.",
  "endpoint_url": "https://fx7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "data-fetch",
    "data-transform"
  ],
  "input_schema": {
    "from": {
      "type": "string",
      "required": true
    },
    "to": {
      "type": "string",
      "required": true
    },
    "amount": {
      "type": "number",
      "required": true
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "from": {
        "type": "string"
      },
      "to": {
        "type": "string"
      },
      "amount": {
        "type": "number"
      },
      "rate": {
        "type": "number"
      },
      "converted": {
        "type": "number"
      }
    }
  },
  "adk_version": "1.1.1",
  "llm_base_url": "http://localhost:11500",
  "dag": [
    {
      "name": "fetch",
      "depends_on": [],
      "description": "Fetch rate from open.er-api.com",
      "capability": null
    },
    {
      "name": "convert",
      "depends_on": [
        "fetch"
      ],
      "description": "Apply conversion formula",
      "capability": null
    }
  ]
}
```

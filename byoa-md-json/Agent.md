# Agent: byoa-md-json

Converts a markdown document (e.g. OCR output) into a single structured JSON object, with row-level table extraction and repair/verification passes.

```json
{
  "agent_id": "byoa-md-json",
  "display_name": "byoa-md-json",
  "version": "1.0.0",
  "description": "Converts a markdown document (e.g. OCR output) into a single structured JSON object, with row-level table extraction and repair/verification passes.",
  "endpoint_url": "https://md-json.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "text-extract"
  ],
  "input_schema": {
    "md": {
      "type": "string",
      "required": true
    },
    "focus": {
      "type": "string",
      "required": false
    },
    "focus_rules": {
      "type": "string",
      "required": false
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "string"
      },
      "structured": {
        "type": "object",
        "nullable": true
      },
      "review_metadata": {
        "type": "object",
        "required": false
      },
      "output_format": {
        "type": "string"
      },
      "json_parse_failed": {
        "type": "boolean"
      },
      "json_repaired": {
        "type": "boolean"
      },
      "json_meta_refusal_retried": {
        "type": "boolean"
      },
      "input_length": {
        "type": "integer"
      },
      "model": {
        "type": "string"
      },
      "extraction_mode": {
        "type": "string",
        "required": false
      },
      "row_count": {
        "type": "integer",
        "required": false
      }
    }
  },
  "adk_version": "1.1.1",
  "llm_base_url": null,
  "dag": [
    {
      "name": "validate",
      "depends_on": [],
      "description": "Validate inputs.md",
      "capability": null
    },
    {
      "name": "extract",
      "depends_on": [
        "validate"
      ],
      "description": "Extract JSON from markdown via vLLM",
      "capability": null
    }
  ]
}
```

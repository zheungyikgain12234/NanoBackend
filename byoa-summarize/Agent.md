# Agent: byoa-summarize-agent

Accepts plain text and returns a 4-6 sentence summary with good detail using Ollama. Stateless, GPU-accelerated.

```json
{
  "agent_id": "byoa-summarize-agent",
  "display_name": "byoa-summarize-agent",
  "version": "2.1.0",
  "description": "Accepts plain text and returns a 4-6 sentence summary with good detail using Ollama. Stateless, GPU-accelerated.",
  "endpoint_url": "https://summarize7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "text-summarize"
  ],
  "input_schema": {
    "text": {
      "type": "string",
      "required": true
    },
    "focus": {
      "type": "string",
      "required": false
    },
    "output_format": {
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
      "input_length": {
        "type": "integer"
      },
      "model": {
        "type": "string"
      },
      "gap_fill_triggered": {
        "type": "boolean"
      },
      "extraction_condensed": {
        "type": "boolean"
      },
      "extraction_echo_retried": {
        "type": "boolean"
      },
      "extraction_type_corrected": {
        "type": "boolean"
      },
      "list_reformatted": {
        "type": "boolean"
      },
      "structured": {
        "type": [
          "object",
          "null"
        ]
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
      "doc_id": {
        "type": [
          "string",
          "null"
        ]
      },
      "api_doc_type": {
        "type": [
          "string",
          "null"
        ]
      },
      "ap_submission": {
        "type": [
          "object",
          "null"
        ]
      }
    }
  },
  "adk_version": "1.1.1",
  "llm_base_url": null,
  "dag": [
    {
      "name": "validate",
      "depends_on": [],
      "description": "Validate inputs.text",
      "capability": null
    },
    {
      "name": "summarize",
      "depends_on": [
        "validate"
      ],
      "description": "Call Ollama",
      "capability": null
    }
  ]
}
```

# Agent: byoa-val-json

Compares an actual JSON against a ground-truth JSON by value, tolerating different field names, nesting and ordering.

```json
{
  "agent_id": "byoa-val-json",
  "display_name": "byoa-val-json",
  "version": "1.0.0",
  "description": "Compares an actual JSON against a ground-truth JSON by value, tolerating different field names, nesting and ordering.",
  "endpoint_url": "https://val-json.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "data-matching"
  ],
  "input_schema": {
    "actual": {
      "type": "object",
      "required": true
    },
    "ground_truth": {
      "type": "object",
      "required": true
    },
    "numeric_tolerance": {
      "type": "number",
      "required": false
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "match": {
        "type": "boolean"
      },
      "score": {
        "type": "number"
      },
      "total_ground_truth_fields": {
        "type": "integer"
      },
      "matched_count": {
        "type": "integer"
      },
      "matches": {
        "type": "array"
      },
      "mismatches": {
        "type": "array"
      },
      "missing": {
        "type": "array"
      },
      "extra_fields": {
        "type": "array"
      },
      "llm_used": {
        "type": "boolean"
      },
      "llm_error": {
        "type": "string"
      },
      "all_important_fields_match": {
        "type": "boolean",
        "nullable": true
      },
      "summary": {
        "type": "string"
      }
    }
  },
  "adk_version": "1.1.1",
  "llm_base_url": "http://localhost:11500"
}
```

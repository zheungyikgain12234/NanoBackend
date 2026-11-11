# Agent: byoa-groundtruth

Fetches the ground-truth JSON for a document from the DocStore API by document number. Pass the ENTIRE extracted JSON as `actual` (e.g. $task_1.structured); the agent finds the document number itself, whatever the field is called.

```json
{
  "agent_id": "byoa-groundtruth",
  "display_name": "byoa-groundtruth",
  "version": "1.0.0",
  "description": "Fetches the ground-truth JSON for a document from the DocStore API by document number. Pass the ENTIRE extracted JSON as `actual` (e.g. $task_1.structured); the agent finds the document number itself, whatever the field is called.",
  "endpoint_url": "https://groundtruth.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "data-fetch"
  ],
  "input_schema": {
    "actual": {
      "type": "object",
      "required": false,
      "description": "The whole extracted JSON (pass it entirely, not one field). The document number is located inside it."
    },
    "document_no": {
      "type": "string",
      "required": false,
      "description": "Only if the number is already known exactly."
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "found": {
        "type": "boolean"
      },
      "document_no": {
        "type": "string"
      },
      "ground_truth": {
        "type": "object"
      },
      "source_file": {
        "type": "string"
      },
      "tried": {
        "type": "array"
      },
      "method": {
        "type": "string"
      }
    }
  },
  "adk_version": "1.1.1",
  "llm_base_url": "http://localhost:11500"
}
```

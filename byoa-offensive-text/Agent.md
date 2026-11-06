# Agent: offensive-text-agent

Classifies input text as offensive or not offensive using an LLM.

```json
{
  "agent_id": "offensive-text-agent",
  "display_name": "offensive-text-agent",
  "version": "1.0.0",
  "description": "Classifies input text as offensive or not offensive using an LLM.",
  "endpoint_url": "https://offensive-text7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "text-classify"
  ],
  "input_schema": {
    "text": {
      "type": "string",
      "required": true
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "is_offensive": {
        "type": "boolean"
      },
      "confidence": {
        "type": "number"
      },
      "reason": {
        "type": "string"
      }
    }
  },
  "adk_version": "1.0.0",
  "llm_base_url": "http://localhost:11500"
}
```

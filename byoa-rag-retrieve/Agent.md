# Agent: byoa-knowledge-retrieve-agent

Searches whatever documents have been uploaded into this platform's internal knowledge base and returns the most relevant passages for a query -- covering any topic those documents contain, not just company/business content. Useful for any factual or informational question where the answer might already be sitting in an ingested document, before falling back to a live web search. Accepts an optional session_id to scope the search to only documents uploaded under that session (e.g. files attached to the current goal), rather than the entire knowledge base. Also accepts an optional fetch_all flag (requires session_id) that bypasses similarity search entirely and returns the ENTIRE session's content, in original document order, as a single full_text field -- use this instead of a normal similarity query for goals asking to summarize/describe a WHOLE uploaded document, since a query like 'summarize the document' has little semantic overlap with the document's actual content and will otherwise score low or trigger a false fallback recommendation. Reports a fallback_recommended flag (with a plain-language reason) when nothing found is a confident match (or, in fetch_all mode, when nothing has been ingested under that session_id yet), so the caller knows to try elsewhere and can explain why.

```json
{
  "agent_id": "byoa-knowledge-retrieve-agent",
  "display_name": "byoa-knowledge-retrieve-agent",
  "version": "1.0.0",
  "description": "Searches whatever documents have been uploaded into this platform's internal knowledge base and returns the most relevant passages for a query -- covering any topic those documents contain, not just company/business content. Useful for any factual or informational question where the answer might already be sitting in an ingested document, before falling back to a live web search. Accepts an optional session_id to scope the search to only documents uploaded under that session (e.g. files attached to the current goal), rather than the entire knowledge base. Also accepts an optional fetch_all flag (requires session_id) that bypasses similarity search entirely and returns the ENTIRE session's content, in original document order, as a single full_text field -- use this instead of a normal similarity query for goals asking to summarize/describe a WHOLE uploaded document, since a query like 'summarize the document' has little semantic overlap with the document's actual content and will otherwise score low or trigger a false fallback recommendation. Reports a fallback_recommended flag (with a plain-language reason) when nothing found is a confident match (or, in fetch_all mode, when nothing has been ingested under that session_id yet), so the caller knows to try elsewhere and can explain why.",
  "endpoint_url": "https://retrieve7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "knowledge-retrieve"
  ],
  "input_schema": {
    "query": {
      "type": "string",
      "required": true
    },
    "top_k": {
      "type": "integer",
      "required": false
    },
    "min_similarity": {
      "type": "number",
      "required": false
    },
    "session_id": {
      "type": "string",
      "required": false
    },
    "fetch_all": {
      "type": "boolean",
      "required": false
    },
    "doc_id": {
      "type": "string",
      "required": false
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "nullable": true
      },
      "results": {
        "type": "array"
      },
      "full_text": {
        "type": "string",
        "required": false
      },
      "truncated": {
        "type": "boolean",
        "required": false
      },
      "best_similarity": {
        "type": "number",
        "nullable": true
      },
      "fallback_recommended": {
        "type": "boolean"
      },
      "fallback_reason": {
        "type": "string",
        "nullable": true
      }
    }
  },
  "adk_version": "1.0.0",
  "llm_base_url": null,
  "dag": [
    {
      "name": "embed_query",
      "depends_on": [],
      "description": "Embed the query text",
      "capability": null
    },
    {
      "name": "search",
      "depends_on": [
        "embed_query"
      ],
      "description": "Query ChromaDB for nearest chunks",
      "capability": null
    },
    {
      "name": "assess",
      "depends_on": [
        "search"
      ],
      "description": "Decide fallback recommendation",
      "capability": null
    }
  ]
}
```

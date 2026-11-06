# Agent: nanowaka-fin-ocr-agent

Converts PDF files (one, several, or a whole directory) to Markdown with the nanowaka.fin.ocr model, an OvisOCR2 finetune trained on financial documents (invoices, purchase orders, delivery orders). Input: base64 PDFs in `files`, or server-side `paths` / `directory`. Output: one Markdown document per PDF, pages separated by '---', plus quality flags (runaway output, malformed tables).

```json
{
  "agent_id": "nanowaka-fin-ocr-agent",
  "display_name": "nanowaka-fin-ocr-agent",
  "version": "1.0.0",
  "description": "Converts PDF files (one, several, or a whole directory) to Markdown with the nanowaka.fin.ocr model, an OvisOCR2 finetune trained on financial documents (invoices, purchase orders, delivery orders). Input: base64 PDFs in `files`, or server-side `paths` / `directory`. Output: one Markdown document per PDF, pages separated by '---', plus quality flags (runaway output, malformed tables).",
  "endpoint_url": "https://ocr7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "text-extract"
  ],
  "input_schema": {
    "files": {
      "type": "array",
      "required": false,
      "description": "[{filename, content_base64}] -- PDFs sent inline as base64."
    },
    "paths": {
      "type": "array",
      "required": false,
      "description": "Server-side PDF file or directory paths (inside OCR_ALLOWED_INPUT_ROOTS)."
    },
    "directory": {
      "type": "string",
      "required": false,
      "description": "Server-side directory; searched recursively for *.pdf."
    },
    "dpi": {
      "type": "integer",
      "required": false,
      "description": "Render DPI. Default 200 (matches training)."
    },
    "max_new_tokens": {
      "type": "integer",
      "required": false,
      "description": "Per-page token limit. Default 4096."
    },
    "per_file_markdown": {
      "type": "boolean",
      "required": false,
      "description": "true = also return each PDF's Markdown in results[].markdown (default false: the Markdown is returned once, in `markdown`)."
    },
    "include_markdown": {
      "type": "boolean",
      "required": false,
      "description": "false = return only output_path/flags, not the Markdown text."
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "model": {
        "type": "string"
      },
      "markdown": {
        "type": "string",
        "description": "All Markdown in one string (several PDFs are joined, each under an '<!-- file: name -->' marker). Same text as results[].markdown."
      },
      "summary": {
        "type": "object",
        "description": "files/ok/flagged/errors/pages/output_dir"
      },
      "results": {
        "type": "array",
        "description": "per PDF: filename, status (ok|flagged|error), pages, markdown, flags, output_path, seconds"
      }
    }
  },
  "adk_version": "1.1.1",
  "llm_base_url": null
}
```

# Agent: json-match-agent

Matches qty/amount data across 2+ OCR-extracted JSON documents with differing field names (e.g. Invoice/PO/DO 3-way matching). Uses AI for field alignment, deterministic checks for numeric comparison. By default only line items are compared -- header-level monetary fields (GST, subtotal, grand total, etc.) are ignored; pass "line_items_only": false to also compare those. The first document in the list is treated as the reference; all others are compared against it. Documents sharing an identical schema (e.g. a shipment split across two GRNs) are auto-grouped and summed before comparison; pass "groups" to override. PO-vs-Invoice/GRN comparisons are treated as a lenient "commitment check" (a lower quantity/amount than the PO is a normal open-order status, not a discrepancy; a higher one still is), while Invoice-vs-GRN comparisons stay a strict "fulfillment check". Also supports credit-note netting ("credit_note_indices"/"credit_note_targets") to subtract a credit note's amounts from its target document before comparison. When `documents` contains a PO-role document alongside at least one INVOICE-role document, the response shape changes automatically: instead of one "comparisons" list, you get a chronological "final_summary"/"timeline" -- each invoice is checked against what was still left on the PO and against only the GRNs already dated on or before that invoice's own date (a GRN dated after it wasn't there yet, so billing goods before their GRN arrives is correctly flagged even for a single invoice); earlier invoices' quantities are netted out first, so a line already fully invoiced is correctly flagged DISCREPANCY -- double invoicing -- if billed again, rather than read as a normal partial fulfillment. "invoice_under_review_index" optionally names which invoice's result should be surfaced at the top level ("final_overall_match"/"final_status"); it defaults to the most recent invoice by date. With no PO present at all (e.g. a plain Invoice-vs-GRN or PO-vs-GRN call), the response is unchanged from before this existed. Large uploads (more than PARALLEL_MATCH_MIN_DOCS documents, default 5) are matched in parallel: when every document carries a PO reference and they form two or more separate PO transactions, each transaction is matched as its own batch (see "batches"), batches running in parallel; otherwise the comparisons inside the one match run in parallel.

```json
{
  "agent_id": "json-match-agent",
  "display_name": "json-match-agent",
  "version": "1.7.0",
  "description": "Matches qty/amount data across 2+ OCR-extracted JSON documents with differing field names (e.g. Invoice/PO/DO 3-way matching). Uses AI for field alignment, deterministic checks for numeric comparison. By default only line items are compared -- header-level monetary fields (GST, subtotal, grand total, etc.) are ignored; pass \"line_items_only\": false to also compare those. The first document in the list is treated as the reference; all others are compared against it. Documents sharing an identical schema (e.g. a shipment split across two GRNs) are auto-grouped and summed before comparison; pass \"groups\" to override. PO-vs-Invoice/GRN comparisons are treated as a lenient \"commitment check\" (a lower quantity/amount than the PO is a normal open-order status, not a discrepancy; a higher one still is), while Invoice-vs-GRN comparisons stay a strict \"fulfillment check\". Also supports credit-note netting (\"credit_note_indices\"/\"credit_note_targets\") to subtract a credit note's amounts from its target document before comparison. When `documents` contains a PO-role document alongside at least one INVOICE-role document, the response shape changes automatically: instead of one \"comparisons\" list, you get a chronological \"final_summary\"/\"timeline\" -- each invoice is checked against what was still left on the PO and against only the GRNs already dated on or before that invoice's own date (a GRN dated after it wasn't there yet, so billing goods before their GRN arrives is correctly flagged even for a single invoice); earlier invoices' quantities are netted out first, so a line already fully invoiced is correctly flagged DISCREPANCY -- double invoicing -- if billed again, rather than read as a normal partial fulfillment. \"invoice_under_review_index\" optionally names which invoice's result should be surfaced at the top level (\"final_overall_match\"/\"final_status\"); it defaults to the most recent invoice by date. With no PO present at all (e.g. a plain Invoice-vs-GRN or PO-vs-GRN call), the response is unchanged from before this existed. Large uploads (more than PARALLEL_MATCH_MIN_DOCS documents, default 5) are matched in parallel: when every document carries a PO reference and they form two or more separate PO transactions, each transaction is matched as its own batch (see \"batches\"), batches running in parallel; otherwise the comparisons inside the one match run in parallel.",
  "endpoint_url": "https://match7.zackloh.dev",
  "health_check_path": "/health",
  "invoke_path": "/run",
  "invoke_method": "POST",
  "invoke_content_type": "application/json",
  "capabilities": [
    "data-matching"
  ],
  "input_schema": {
    "documents": {
      "type": "array",
      "required": true,
      "minItems": 2
    },
    "amount_tolerance": {
      "type": "string",
      "required": false
    },
    "line_items_only": {
      "type": "boolean",
      "required": false,
      "description": "IGNORED in this build -- always true (forced in run()). When true, only line items decide the match/status -- header-level monetary fields (GST, subtotal, grand total, shipping, etc.) are never compared and never appear in the output. Set to false to also compare those header fields, as in versions before this flag."
    },
    "groups": {
      "type": "array",
      "required": false,
      "description": "List of index-lists into `documents`. Each inner list's documents are merged (qty/amount fields summed) into ONE comparison target before matching -- e.g. [[2, 3]] merges documents[2] and documents[3] (two partial GRNs for one shipment) into a single combined GRN. OPTIONAL: when omitted, documents sharing an identical schema (same field names) are auto-grouped this way automatically -- pass \"groups\": [] explicitly to force the old no-auto-grouping behavior instead. Groups are ordered by their smallest member index, so the existing 'first document is the reference' convention extends to 'first group is the reference'."
    },
    "credit_note_indices": {
      "type": "array",
      "required": false,
      "description": "List of indices into `documents` identifying which are credit notes. Each credit note's qty/amount fields are SUBTRACTED from its target document (see credit_note_targets) rather than being compared as an independent document. If omitted, credit notes are auto-detected via a document_type/doc_type/type field containing 'credit' -- auto-detection only works for document shapes that carry such a field."
    },
    "credit_note_targets": {
      "type": "object",
      "required": false,
      "description": "Optional {credit_note_index: target_index} map (both indices into `documents`) controlling which document each credit note nets against. A credit note not listed here defaults to netting against whichever document has an INVOICE role (regardless of upload order) -- falling back to the reference document only if no Invoice is present among the documents being compared."
    },
    "invoice_under_review_index": {
      "type": "integer",
      "required": false,
      "description": "Index into `documents` of the invoice being decided on right now. Required when `documents` contains more than one INVOICE-role document (e.g. a PO invoiced in more than one batch over time) -- omitted with exactly one INVOICE-role document, that one is used automatically. Every OTHER INVOICE-role document is treated as a prior invoice already recorded against this PO: its qty/amount fields are subtracted from every PO and GRN document/group before the invoice under review is compared against them, so a line already fully invoiced earlier is flagged DISCREPANCY (double invoicing) if billed again, instead of looking like a normal open order."
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "final_overall_match": {
        "type": "boolean",
        "description": "The headline verdict for this whole matching session. True unless at least one comparison has status DISCREPANCY. A PARTIAL_FULFILLMENT status (a PO not yet fully delivered/invoiced) does NOT make this false -- that's an expected open-order state, not an error."
      },
      "final_status": {
        "type": "string",
        "description": "The headline status for this whole matching session: CONSISTENT, PARTIAL_FULFILLMENT, DISCREPANCY, or NOT_APPLICABLE. In the multi-invoice response, this is whichever invoice's status is surfaced as current (see current_invoice_index) -- not necessarily the most severe one across the whole timeline."
      },
      "final_label": {
        "type": [
          "string",
          "null"
        ],
        "description": "The headline label for this whole matching session, present alongside final_overall_match/final_status. \"MATCH\" when final_overall_match is true; when final_status is DISCREPANCY, one of: \"QTY_MISMATCH\" (a quantity disagreement, no amount problem), \"AMOUNT_MISMATCH\" (amounts disagree but quantities line up), \"QTY_AND_AMOUNT_MISMATCH\" (both -- either the same line has both problems, or different lines have different ones), or \"OTHER_MISMATCH\" (a DISCREPANCY not tied to a specific qty/amount line comparison at all, e.g. a role mismatch or missing-data-only cause). Combines the union of every underlying comparison: any QTY plus any AMOUNT anywhere makes this QTY_AND_AMOUNT_MISMATCH even if no single line had both. Each entry inside \"comparisons\" and \"timeline\" also carries its OWN, separate \"label\" -- that one describes just that one entry, not the session."
      },
      "currency": {
        "type": "string",
        "description": "Present alongside \"final_label\" wherever it appears. Whichever currency a document in the comparison stated explicitly (a \"currency\"/\"currency_code\" field, e.g. \"USD\", \"SGD\"), or \"S$\" if none of them said. Display-only -- shown as a prefix on dollar amounts inside \"final_summary\" text (e.g. \"S$4.00\", \"USD 4.00\"); never affects matching logic, and the numeric amount_a/amount_b fields elsewhere in the output stay plain, unprefixed decimal strings."
      },
      "final_summary": {
        "type": "string",
        "description": "The headline, human-readable readout for this whole matching session -- this is the field to display as the main match result. In the single-comparison-set response (no \"timeline\"), it's a combined plain-English readout of every comparison, e.g. 'Invoice matches GRN/Delivery -- fully consistent. PO shows partial fulfillment: 3870 of 20000 ordered units received/invoiced -- not a discrepancy, an open-order status.' In the multi-invoice response (see \"timeline\"), it's instead a line-per-invoice narrative in chronological order, e.g. 'In INV1 (2026-01-22), ... partial fulfillment ...\\nIn INV2 (2026-06-02), ... discrepancy ...', joined from each timeline entry's own \"narrative\"."
      },
      "reference_index": {
        "type": "integer"
      },
      "document_count": {
        "type": "integer"
      },
      "document_roles": {
        "type": "object",
        "description": "{document_index: role}, role one of PO, INVOICE, GRN, CREDIT_NOTE, UNKNOWN -- best-effort, used to decide which comparisons are lenient (commitment_check, PO vs Invoice/GRN) vs strict (fulfillment_check, Invoice vs GRN)."
      },
      "document_ids": {
        "type": "object",
        "description": "{document_index: doc_id}, each document's own identifying number (invoice no., PO no., GRN no., etc.), e.g. \"PPINV-B10097157\" -- best-effort, null when no self-identifying field is found. Display-only, never affects matching logic."
      },
      "comparisons": {
        "type": "array",
        "description": "One entry per non-reference document, each with overall_match, status (CONSISTENT/PARTIAL_FULFILLMENT/DISCREPANCY/NOT_APPLICABLE), mode (fulfillment_check/commitment_check/generic), doc_a_role, doc_b_role, doc_a_id, doc_b_id, summary (plain-English one-liner), monetary_fields (list, each now also carrying a status), line_items (list, each now also carrying qty_status/amount_status), unmatched_doc_a_indices, unmatched_doc_b_indices, excluded_doc_a_rows, excluded_doc_b_rows. Only present when `documents` had no PO-role document alongside an INVOICE-role one -- see \"timeline\" below for that response shape instead."
      },
      "current_invoice_index": {
        "type": "integer",
        "description": "Only present in the multi-invoice response (see \"timeline\"). Index into `documents` of whichever invoice's result is surfaced in the top-level final_overall_match/final_status -- either invoice_under_review_index, or the most recent invoice by date if that wasn't given."
      },
      "timeline": {
        "type": "array",
        "description": "Only present in the multi-invoice response: one entry per INVOICE-role document. Ordered by each invoice's own date field ONLY if every invoice in `documents` has one that parses; if even one is missing/unparseable, all entries instead keep the order their documents were given in (date_confidence flags which applies per entry) -- partial date coverage is never used to reorder, since guessing at an undated invoice's position would also corrupt the running PO/GRN netting for every invoice after it. Each entry has document_index, invoice_id, date (ISO string, or \"date unknown\"), date_confidence (\"parsed\"/\"unknown_used_document_order\"), status, overall_match, narrative (one-line human-readable verdict for this invoice alone), and comparisons (this invoice's own PO/GRN comparisons, in the same shape as the single-invoice \"comparisons\" field, checked against what was still left on the PO/GRN after every earlier invoice in the timeline had already been netted out). An invoice document whose own id starts with \"P\" (e.g. \"PPINV-B10097417\") never gets its own entry here when at least one other, non-\"P\" invoice is also present -- see \"posted_invoices_excluded\" and \"already_paid\"."
      },
      "batch_count": {
        "type": "integer",
        "description": "Only present when a large upload was split into independent PO transactions and matched in parallel batches."
      },
      "batches": {
        "type": "array",
        "description": "Only present together with batch_count: one entry per transaction, each with batch_key (its PO number), document_indices (positions in `documents`), index_scope (\"batch_local\": document indices INSIDE an entry refer to that batch's own document order, map them through document_indices), and that transaction's full normal result (final_status, comparisons or timeline, ...) -- or an \"error\" string if that one batch could not be matched, which counts as a DISCREPANCY in the top-level verdict."
      },
      "posted_invoices_excluded": {
        "type": "array",
        "description": "Only present when `documents` contained BOTH an invoice whose own id starts with \"P\" (a posted/system record, e.g. \"PPINV-B10097417\") and at least one that doesn't (the vendor's original invoice, e.g. \"SI1202997\"). Lists the excluded posted invoice(s)' own doc_id(s) -- they are never compared against the PO/GRN or given their own \"timeline\" entry, since they represent the SAME purchase as the vendor invoice already posted/paid, not a second one; comparing both would double-consume the PO's quantities. They're used only to populate \"already_paid\" (see \"comparisons\" and \"timeline\"). If a posted invoice is the only invoice present, this field is absent and it's matched normally like any other invoice."
      }
    }
  },
  "adk_version": "1.1.1",
  "llm_base_url": null
}
```

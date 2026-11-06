"""
Model + PDF handling for nanowaka-ocr-agent.

Loads the fine-tuned OCR model from Hugging Face ONCE (lazily, on the first
request) and exposes ocr_pdf_bytes() -> (markdown, pages, problems).
GPU generation is serialised behind one lock, so concurrent requests queue
instead of fighting over GPU memory.
"""
import os
import re
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("OCR_MODEL_ID", "zheungyikgain/nanowaka.fin.ocr")
FALLBACK_PROCESSOR = os.environ.get("OCR_FALLBACK_PROCESSOR", "ATH-MaaS/OvisOCR2")
PROMPT = (HERE / "prompt.txt").read_text(encoding="utf-8")   # same prompt as training
DEFAULT_DPI = int(os.environ.get("OCR_DPI", "200"))           # same render as training
DEFAULT_MAX_NEW = int(os.environ.get("OCR_MAX_NEW_TOKENS", "4096"))
MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES_PER_PDF", "200"))

_load_lock = threading.Lock()
_gen_lock = threading.Lock()
_state = {}


def _load():
    with _load_lock:
        if _state:
            return _state
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID)
        except Exception:
            processor = AutoProcessor.from_pretrained(FALLBACK_PROCESSOR)
        model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
        tok = getattr(processor, "tokenizer", processor)
        stop_ids = sorted({i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
                           if isinstance(i, int) and i >= 0})
        _state.update(torch=torch, processor=processor, model=model, stop_ids=stop_ids)
        return _state


def is_loaded() -> bool:
    return bool(_state)


def _ocr_image(img, max_new):
    s = _load()
    torch, processor, model = s["torch"], s["processor"], s["model"]
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False).to(model.device)
    with _gen_lock, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, eos_token_id=s["stop_ids"])
    n_in = inputs["input_ids"].shape[-1]
    text = processor.decode(out[0][n_in:], skip_special_tokens=True).split("<think>")[0].strip()
    return text, out.shape[-1] - n_in


def render_pdf(pdf_bytes: bytes, dpi: int):
    import pymupdf
    from PIL import Image
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count > MAX_PAGES:
            raise ValueError(f"PDF has {doc.page_count} pages; limit is {MAX_PAGES}")
        imgs = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return imgs
    finally:
        doc.close()


def ocr_pdf_bytes(pdf_bytes: bytes, dpi: int = DEFAULT_DPI, max_new: int = DEFAULT_MAX_NEW):
    """Returns (markdown, page_count, problems). Pages are joined with a '---' rule."""
    texts, reasons = [], []
    imgs = render_pdf(pdf_bytes, dpi)
    for k, img in enumerate(imgs, 1):
        text, ntok = _ocr_image(img, max_new)
        texts.append(text)
        reasons += [f"page {k}: {r}" for r in problems(text, ntok, max_new)]
    return "\n\n---\n\n".join(texts), len(imgs), reasons


# --- output sanity checks (same rules as ocr_folder.py / lint.py) -----------

def _cells(line):
    s = line.strip()
    if len(s) < 2 or not (s.startswith("|") and s.endswith("|")):
        return None
    return [c.strip() for c in s[1:-1].split("|")]


def _is_sep(cs):
    return all(re.fullmatch(r":?-{3,}:?", c) for c in cs)


def problems(text, ntok, max_new):
    out = []
    if ntok >= max_new:
        out.append(f"hit the {max_new}-token limit (runaway)")
    if not text.strip():
        out.append("empty output")
    if "<table" in text.lower():
        out.append("HTML table in output")
    lines, i = text.split("\n"), 0
    while i < len(lines):
        if _cells(lines[i]) is None:
            i += 1
            continue
        j = i
        while j < len(lines) and _cells(lines[j]) is not None:
            j += 1
        block = [_cells(l) for l in lines[i:j]]
        width, run, maxrun = len(block[0]), 0, 0
        for r in block:
            run = run + 1 if (all(c == "" for c in r) and not _is_sep(r)) else 0
            maxrun = max(maxrun, run)
        if maxrun >= 5:
            out.append(f"table with {maxrun} empty rows in a row")
        bad = [r for r in block if len(r) != width]
        if bad:
            out.append(f"{len(bad)} table row(s) with a different cell count than the header ({width})")
        i = j
    return out

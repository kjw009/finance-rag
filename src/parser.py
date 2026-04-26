from __future__ import annotations

import re
import sys
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import Chunk, make_chunk_id

CHUNK_SIZE = 800
OVERLAP = 80
MIN_CHARS = 40
CLASSIFY_PAGES = 3
SLIDE_CHARS_THRESHOLD = 1500

def filing_date_from_filename(fname: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    return m.group(1) if m else None

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def diagnose(path: Path) -> dict:
    total_chars = total_tables = page_count = 0
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            total_chars += len(page.extract_text() or "")
            total_tables += len(page.find_tables())
    chars_per_page = total_chars / max(page_count, 1)
    return {
        "file": path.name,
        "pages": page_count,
        "chars_per_page": round(chars_per_page, 1),
        "tables": total_tables,
        "likely_scan": chars_per_page < 100,
    }


def classify(path: Path, diag: dict) -> tuple[str, str]:
    if diag["chars_per_page"] < SLIDE_CHARS_THRESHOLD:
        return "earnings_slides", "per_page"
    sample = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:CLASSIFY_PAGES]:
            sample += (page.extract_text() or "").lower()
    if "form 10-k" in sample or "annual report" in sample:
        return "10-K", "overlapping"
    if "form 10-q" in sample:
        return "10-Q", "overlapping"
    if "exhibit 99" in sample:
        return "earnings_release", "overlapping"
    if diag["tables"] / max(diag["pages"], 1) > 1.0:
        return "financial_report", "overlapping"
    return "unknown", "overlapping"


def extract_prose_text(page) -> str:
    table_bboxes = [t.bbox for t in page.find_tables()]
    if not table_bboxes:
        return page.extract_text() or ""

    def not_in_table(obj):
        for x0, top, x1, bottom in table_bboxes:
            if x0 - 1 <= obj["x0"] and obj["x1"] <= x1 + 1 and \
               top - 1 <= obj["top"] and obj["bottom"] <= bottom + 1:
                return False
        return True

    return page.filter(not_in_table).extract_text() or ""


# ---------------------------------------------------------------------------
# HTM helpers
# ---------------------------------------------------------------------------

def classify_htm(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    if len(parts) >= 2:
        form = parts[1].upper()
        if form in ("10-K", "10-Q", "8-K"):
            return form, "overlapping"
    return "unknown", "overlapping"


def extract_htm_table(table_elem) -> list[list[str]]:
    rows = []
    for tr in table_elem.find_all("tr", recursive=False):
        cells = [
            cell.get_text(separator=" ", strip=True)
            for cell in tr.find_all(["th", "td"], recursive=False)
        ]
        if any(cells):
            rows.append(cells)
    return rows


# ---------------------------------------------------------------------------
# Shared chunking
# ---------------------------------------------------------------------------

def table_to_chunk(
    table_data: list[list],
    page_num: int | None,
    source: str,
    idx: int,
    doc_type: str,
    filing_date: str | None = None,
    extra: dict | None = None,
) -> Chunk | None:
    rows = [[str(c or "").strip() for c in row] for row in table_data]
    rows = [r for r in rows if any(r)]
    if not rows:
        return None
    header = rows[0]
    sep = "| " + " | ".join("---" for _ in header) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    md = "\n".join(lines)
    chunk = Chunk(
        id=make_chunk_id(source, idx, md),
        text=md,
        source=source,
        page=page_num,
        content_type="table",
        document_type=doc_type,
        filing_date=filing_date,
    )
    if extra:
        chunk.extra.update(extra)
    return chunk


def prose_to_chunks(
    text: str,
    page_num: int | None,
    source: str,
    start_idx: int,
    doc_type: str,
    filing_date: str | None = None,
    extra: dict | None = None,
) -> list[Chunk]:
    text = text.strip()
    if len(text) < MIN_CHARS:
        return []
    chunks, idx, pos = [], start_idx, 0
    while pos < len(text):
        snippet = text[pos : pos + CHUNK_SIZE]
        if len(snippet) >= MIN_CHARS:
            chunk = Chunk(
                id=make_chunk_id(source, idx, snippet),
                text=snippet,
                source=source,
                page=page_num,
                content_type="prose",
                document_type=doc_type,
                filing_date=filing_date,
            )
            if extra:
                chunk.extra.update(extra)
            chunks.append(chunk)
            idx += 1
        pos += CHUNK_SIZE - OVERLAP
    return chunks


def page_to_chunk(
    page, page_num: int, source: str, idx: int, doc_type: str, filing_date: str | None = None
) -> Chunk | None:
    text = (page.extract_text() or "").strip()
    if len(text) < MIN_CHARS:
        return None
    return Chunk(
        id=make_chunk_id(source, idx, text),
        text=text,
        source=source,
        page=page_num,
        content_type="prose",
        document_type=doc_type,
        filing_date=filing_date,
    )


# ---------------------------------------------------------------------------
# Per-file parsers
# ---------------------------------------------------------------------------

def _extra_from_stem(stem: str) -> dict:
    """Extract ticker from TICKER_FORM_DATE filename stems."""
    parts = stem.split("_")
    return {"ticker": parts[0]} if parts else {}


def parse_pdf(path: Path) -> list[Chunk]:
    source = path.name
    diag = diagnose(path)
    if diag["likely_scan"]:
        print(f"  [SKIP] {source} — likely scan, no text layer")
        return []
    doc_type, mode = classify(path, diag)
    extra = _extra_from_stem(path.stem)
    filing_date = filing_date_from_filename(path.name)
    chunks, idx = [], 0
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if mode == "per_page":
                chunk = page_to_chunk(page, page_num, source, idx, doc_type, filing_date)
                if chunk:
                    chunk.extra.update(extra)
                    chunks.append(chunk)
                    idx += 1
            else:
                for tbl in page.find_tables():
                    chunk = table_to_chunk(tbl.extract(), page_num, source, idx, doc_type,filing_date, extra)
                    if chunk:
                        chunks.append(chunk)
                        idx += 1
                new = prose_to_chunks(extract_prose_text(page), page_num, source, idx, doc_type, filing_date, extra)
                chunks.extend(new)
                idx += len(new)
    return chunks


def parse_htm(path: Path) -> list[Chunk]:
    source = path.name
    doc_type, _ = classify_htm(path)
    filing_date = filing_date_from_filename(path.name)
    extra = _extra_from_stem(path.stem)
    soup = BeautifulSoup(
        path.read_text(encoding="utf-8", errors="replace"), "html.parser"
    )
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    chunks, idx = [], 0
    top_tables = [t for t in soup.find_all("table") if not t.find_parent("table")]
    for tbl in top_tables:
        chunk = table_to_chunk(extract_htm_table(tbl), None, source, idx, doc_type, filing_date, extra)
        if chunk:
            chunks.append(chunk)
            idx += 1
        tbl.decompose()
    lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip()]
    new = prose_to_chunks("\n".join(lines), None, source, idx, doc_type, filing_date, extra)
    chunks.extend(new)
    return chunks


def parse(path: Path) -> list[Chunk]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in (".htm", ".html"):
        return parse_htm(path)
    print(f"  [SKIP] {path.name} — unsupported format '{ext}'")
    return []


def parse_all(data_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in (".pdf", ".htm", ".html"):
            continue
        chunks = parse(path)
        all_chunks.extend(chunks)
        n_t = sum(1 for c in chunks if c.content_type == "table")
        n_p = sum(1 for c in chunks if c.content_type == "prose")
        print(f"  {path.name:<50} {len(chunks):>4} chunks  ({n_t} tables, {n_p} prose)")
    print(f"\nTotal: {len(all_chunks)} chunks from {data_dir}")
    return all_chunks

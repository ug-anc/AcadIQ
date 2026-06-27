"""PDF extraction (DR-01, DR-02, DR-06).

DR-01  Primary text layer: PyMuPDF (`get_text("dict")`) — fast and, crucially,
       exposes font size + bold flags per span, which the chunker uses to detect
       section headings. pdfplumber is reserved for tables only.
DR-02  Tables: pdfplumber `extract_tables()` — zero extra system dependency
       (Camelot needs Ghostscript) and adequate for v1. Camelot/tabula is the
       documented upgrade if borderless/merged-cell tables prove lossy.
DR-06  Scan detection: per page, if extractable text < N chars AND the page has
       images, treat it as scanned and route to pytesseract OCR. OCR degrades
       gracefully (warns) if tesseract isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold


@dataclass
class Line:
    text: str
    page_number: int
    font_size: float = 11.0
    bold: bool = False


@dataclass
class Table:
    page_number: int
    text: str


@dataclass
class ExtractedDoc:
    source_document: str
    lines: list[Line] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)

    @property
    def body_font_size(self) -> float:
        sizes = [ln.font_size for ln in self.lines if ln.text.strip()]
        if not sizes:
            return 11.0
        sizes.sort()
        return sizes[len(sizes) // 2]  # median


def _render_table(rows: list[list[str | None]]) -> str:
    cleaned = [
        [(cell or "").strip().replace("\n", " ") for cell in row] for row in rows
    ]
    return "\n".join(" | ".join(r) for r in cleaned if any(r))


def _ocr_page(page) -> str:
    """OCR a single PyMuPDF page via pytesseract. Returns "" if unavailable."""
    try:
        import io

        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception as exc:  # tesseract missing / Pillow missing
        print(f"  [warn] OCR unavailable for page: {exc}")
        return ""


def extract_pdf(path: str, source_name: str | None = None) -> ExtractedDoc:
    import fitz  # PyMuPDF
    import pdfplumber

    s = get_settings()
    name = source_name or path.rsplit("/", 1)[-1]
    doc = ExtractedDoc(source_document=name)

    fitz_doc = fitz.open(path)
    for page_index in range(len(fitz_doc)):
        page = fitz_doc[page_index]
        page_no = page_index + 1
        raw_text = page.get_text().strip()

        # DR-06 scan detection.
        is_scanned = len(raw_text) < s.SCAN_TEXT_DENSITY_CHARS and bool(
            page.get_images()
        )
        if is_scanned:
            doc.scanned_pages.append(page_no)
            if s.ENABLE_OCR:
                ocr_text = _ocr_page(page)
                for line in ocr_text.splitlines():
                    if line.strip():
                        doc.lines.append(Line(text=line.strip(), page_number=page_no))
            continue

        # Structured extraction with font metadata for heading detection.
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(sp.get("text", "") for sp in spans).strip()
                if not text:
                    continue
                size = max(sp.get("size", 11.0) for sp in spans)
                bold = any(
                    (sp.get("flags", 0) & _BOLD_FLAG)
                    or "bold" in sp.get("font", "").lower()
                    for sp in spans
                )
                doc.lines.append(
                    Line(text=text, page_number=page_no, font_size=size, bold=bold)
                )
    fitz_doc.close()

    # DR-02 tables via pdfplumber.
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            for tbl in page.extract_tables() or []:
                rendered = _render_table(tbl)
                if rendered.strip():
                    doc.tables.append(
                        Table(page_number=page_index + 1, text=rendered)
                    )

    return doc


def extracted_from_text(name: str, text: str, lines_per_page: int = 45) -> ExtractedDoc:
    """Build an ExtractedDoc from plain text (used by the sample-corpus loader).

    Page breaks honor form-feed characters; otherwise pages are split every
    `lines_per_page` non-empty lines so citations carry a plausible page number.
    """
    doc = ExtractedDoc(source_document=name)
    page_no = 1
    count = 0
    for block in text.split("\f"):
        for raw in block.splitlines():
            line = raw.rstrip()
            if not line.strip():
                continue
            doc.lines.append(Line(text=line.strip(), page_number=page_no))
            count += 1
            if count >= lines_per_page:
                page_no += 1
                count = 0
        page_no += 1
        count = 0
    return doc

"""Optional raw-PDF extraction and rendering for ingestion-loss studies.

The benchmark baseline uses supplied contexts. This module writes a separate
raw corpus so parser errors can be measured without changing baseline evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import pdfplumber
import pypdfium2 as pdfium

from t2rag.config import AppConfig

from .database import RegistryDatabase


RenderMode = Literal["none", "weak", "all"]


@dataclass(slots=True)
class RawPdfStats:
    """Counters returned after an optional raw-PDF extraction run."""

    documents: int = 0
    pages: int = 0
    tables: int = 0
    rendered_pages: int = 0
    layout_json_pages: int = 0
    pdf_text_pages: int = 0


def _clean_cell(value: Any) -> str:
    """Normalize one table cell while preserving signs and visible values."""

    if value is None:
        return ""
    return " ".join(str(value).split()).replace("|", "\\|")


def table_to_markdown(table: Sequence[Sequence[Any]]) -> str:
    """Convert a detected table to readable Markdown without dropping cells."""

    rows = [[_clean_cell(cell) for cell in row] for row in table if row]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def render_pdf_page(
    pdf_path: Path,
    *,
    zero_based_page: int,
    output_path: Path,
    dpi: int,
) -> None:
    """Render one page with PDFium for later visual-model inspection."""

    if dpi <= 0:
        raise ValueError("Render DPI must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document.get_page(zero_based_page)
        try:
            bitmap = page.render(scale=dpi / 72.0)
            image = bitmap.to_pil()
            image.save(output_path, format="PNG")
        finally:
            page.close()
    finally:
        document.close()


def extract_raw_pdfs(
    config: AppConfig,
    *,
    limit: int | None = None,
    render_mode: RenderMode = "weak",
) -> RawPdfStats:
    """Extract canonical PDFs into separate raw-page and raw-table tables.

    TAT-DQA ordered JSON blocks take precedence over plain PDF text. Tables are
    still detected from the PDF layout. A rendered PNG is created only for the
    selected rendering mode and never added to the source dataset directory.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive.")
    if render_mode not in {"none", "weak", "all"}:
        raise ValueError("render_mode must be none, weak, or all.")

    stats = RawPdfStats()
    render_root = config.paths.work_root / "raw_pdf_pages"
    with RegistryDatabase(config.paths.registry_db) as database:
        database.initialize()
        documents = list(database.iter_pdf_documents(limit=limit))
        for document_row in documents:
            document_id = str(document_row["document_id"])
            pdf_path = config.paths.dataset_root / str(document_row["relative_path"])
            layout_pages = database.tat_layout_pages(document_id)

            with pdfplumber.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    internal_page = page_index + 1
                    layout_text = layout_pages.get(internal_page, "").strip()
                    if layout_text:
                        source_text = layout_text
                        extraction_method = "tat_layout_json"
                        stats.layout_json_pages += 1
                    else:
                        source_text = (page.extract_text() or "").strip()
                        extraction_method = "pdfplumber_text"
                        stats.pdf_text_pages += 1

                    raw_tables = page.extract_tables() if config.raw_pdf.extract_tables else []
                    tables = [
                        markdown
                        for table in raw_tables
                        if (markdown := table_to_markdown(table)).strip()
                    ]
                    should_render = render_mode == "all" or (
                        render_mode == "weak"
                        and len(source_text) < config.raw_pdf.minimum_text_characters
                    )
                    rendered_path: Path | None = None
                    if should_render:
                        rendered_path = (
                            render_root / document_id / f"page-{internal_page:03d}.png"
                        )
                        render_pdf_page(
                            pdf_path,
                            zero_based_page=page_index,
                            output_path=rendered_path,
                            dpi=config.raw_pdf.render_dpi,
                        )
                        stats.rendered_pages += 1

                    with database.transaction():
                        database.upsert_raw_page(
                            document_id=document_id,
                            internal_page_number=internal_page,
                            extraction_method=extraction_method,
                            source_text=source_text,
                            rendered_image_path=(
                                str(rendered_path) if rendered_path is not None else None
                            ),
                            width_points=float(page.width),
                            height_points=float(page.height),
                        )
                        database.replace_raw_tables(
                            document_id=document_id,
                            internal_page_number=internal_page,
                            tables=tables,
                            extraction_method="pdfplumber_table",
                        )
                    stats.pages += 1
                    stats.tables += len(tables)
            stats.documents += 1
    return stats

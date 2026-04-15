"""
document_loader.py
──────────────────
Handles PDF ingestion for PaperMind.

Responsibilities:
  1. Load a PDF from a file path using PyMuPDF (fitz)
  2. Extract text page-by-page, preserving page numbers
  3. Clean extracted text (strip artefacts, normalise whitespace)
  4. Detect section headings via heuristics
  5. Chunk text with RecursiveCharacterTextSplitter (chunk=500, overlap=100)
  6. Attach metadata: {source, page, section, chunk_index, total_chunks, doc_type}
  7. Return a list of LangChain Document objects ready for embedding
"""

import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.logger import get_logger
from src.exception import DocumentLoadError

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 50
HEADING_MAX_LEN = 80

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Normalise unicode, remove PDF artefacts, collapse excess whitespace."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\x00", "").replace("\uf0b7", "•")
    text = re.sub(r"^[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _extract_section_heading(lines: list[str], line_idx: int) -> Optional[str]:
    """Heuristic: short line followed by blank line = section heading."""
    line = lines[line_idx].strip()
    if not line or len(line) > HEADING_MAX_LEN:
        return None
    if re.match(r"^\d+$", line):
        return None
    if line_idx + 1 < len(lines) and lines[line_idx + 1].strip() == "":
        return line
    return None


def _detect_sections(page_text: str) -> dict[int, str]:
    """Returns {char_offset: heading} for the page."""
    lines   = page_text.splitlines()
    offset  = 0
    sections: dict[int, str] = {}
    for i, line in enumerate(lines):
        heading = _extract_section_heading(lines, i)
        if heading:
            sections[offset] = heading
        offset += len(line) + 1
    return sections


def _nearest_section(char_start: int, sections: dict[int, str]) -> str:
    """Return the section heading at or before char_start."""
    best = "Unknown"
    for offset, heading in sorted(sections.items()):
        if offset <= char_start:
            best = heading
        else:
            break
    return best


# ── Main class ───────────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Loads a PDF and returns LangChain Document objects with metadata.

    Usage
    -----
    loader = DocumentLoader()
    docs   = loader.load("path/to/file.pdf")
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(f"DocumentLoader ready — chunk={chunk_size}, overlap={chunk_overlap}")

    def load(self, pdf_path: str) -> list[Document]:
        """Load a single PDF → list of Document chunks."""
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise DocumentLoadError(f"PDF not found: {pdf_path}", sys)
        if pdf_path.suffix.lower() != ".pdf":
            raise DocumentLoadError(f"Expected .pdf, got: {pdf_path.suffix}", sys)

        logger.info(f"Loading: {pdf_path.name}")
        try:
            raw_pages = self._extract_pages(pdf_path)
        except Exception as e:
            raise DocumentLoadError(e, sys)

        all_docs: list[Document] = []
        for page_num, raw_text in raw_pages:
            all_docs.extend(self._process_page(raw_text, page_num, pdf_path.name))

        logger.info(f"'{pdf_path.name}' → {len(raw_pages)} pages, {len(all_docs)} chunks")
        return all_docs

    def load_multiple(self, pdf_paths: list[str]) -> list[Document]:
        """Load multiple PDFs and return all chunks combined."""
        all_docs: list[Document] = []
        for path in pdf_paths:
            all_docs.extend(self.load(path))
        logger.info(f"Total chunks across {len(pdf_paths)} PDFs: {len(all_docs)}")
        return all_docs

    def _extract_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Extract raw text per page using PyMuPDF."""
        pages = []
        doc   = fitz.open(str(pdf_path))
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            if text and text.strip():
                pages.append((i + 1, text))
            else:
                logger.debug(f"  Page {i+1}: empty, skipping")
        doc.close()
        return pages

    def _process_page(self, raw_text: str, page_num: int, source_name: str) -> list[Document]:
        """Clean → detect sections → chunk → build Documents."""
        cleaned  = _clean_text(raw_text)
        sections = _detect_sections(cleaned)
        chunks   = self.splitter.split_text(cleaned)

        docs        = []
        char_cursor = 0

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_LEN:
                char_cursor += len(chunk)
                continue

            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source":       source_name,
                    "page":         page_num,
                    "section":      _nearest_section(char_cursor, sections),
                    "chunk_index":  idx,
                    "total_chunks": len(chunks),
                    "doc_type":     "pdf",
                }
            ))
            char_cursor += len(chunk)

        return docs
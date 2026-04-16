import base64
import sys
from pathlib import Path

import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.logger import get_logger
from src.exception import ImageIngestionError

logger = get_logger(__name__)

#Constants

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 50

SUPPORTED_FORMATS = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}

VISION_PROMPT = """You are a document digitization assistant.
Extract ALL text content from this image exactly as it appears.
Preserve the structure: headings, paragraphs, tables, bullet points, numbered lists.
Do not summarize, interpret, or add anything.
Just return the raw extracted text."""

#Main class

class ImageLoader:

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.client = anthropic.Anthropic()
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(f"ImageLoader ready — chunk={chunk_size}, overlap={chunk_overlap}")

    def load(self, image_path: str) -> list[Document]:
        """Load a single image → extract text via Claude Vision → return Document chunks."""
        image_path = Path(image_path).resolve()

        if not image_path.exists():
            raise ImageIngestionError(f"Image not found: {image_path}", sys)

        suffix = image_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ImageIngestionError(
                f"Unsupported format: {suffix}. Supported: {list(SUPPORTED_FORMATS.keys())}", sys
            )

        logger.info(f"Processing image: {image_path.name}")

        # Step 1 — encode image to base64
        try:
            image_data  = self._encode_image(image_path)
            media_type  = SUPPORTED_FORMATS[suffix]
        except Exception as e:
            raise ImageIngestionError(f"Failed to encode image: {e}", sys)

        # Step 2 — send to Claude Vision
        try:
            extracted_text = self._extract_text_with_vision(image_data, media_type)
        except Exception as e:
            raise ImageIngestionError(f"Claude Vision extraction failed: {e}", sys)

        if not extracted_text or not extracted_text.strip():
            raise ImageIngestionError(
                f"Claude Vision returned empty text for: {image_path.name}", sys
            )

        logger.info(f"Extracted {len(extracted_text)} characters from {image_path.name}")

        # Step 3 — chunk and build Documents
        docs = self._build_documents(extracted_text, image_path.name)
        logger.info(f"'{image_path.name}' → {len(docs)} chunks")
        return docs

    def load_multiple(self, image_paths: list[str]) -> list[Document]:
        """Load multiple images and return all chunks combined."""
        all_docs: list[Document] = []
        for path in image_paths:
            all_docs.extend(self.load(path))
        logger.info(f"Total chunks across {len(image_paths)} images: {len(all_docs)}")
        return all_docs

    #Internal

    def _encode_image(self, image_path: Path) -> str:
        """Read image file and return base64 encoded string."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _extract_text_with_vision(self, image_data: str, media_type: str) -> str:
        """Send image to Claude Vision and return extracted text."""
        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": media_type,
                                "data":       image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                    ],
                }
            ],
        )
        return message.content[0].text

    def _build_documents(self, text: str, source_name: str) -> list[Document]:
        """Chunk extracted text and wrap in LangChain Documents with metadata."""
        chunks = self.splitter.split_text(text.strip())
        docs   = []

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_LEN:
                continue

            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source":       source_name,
                    "page":         "image",
                    "section":      "image_ocr",
                    "chunk_index":  idx,
                    "total_chunks": len(chunks),
                    "doc_type":     "image",
                }
            ))

        return docs
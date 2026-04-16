import sys
from pathlib import Path

from src.logger import get_logger
from src.exception import DocumentLoadError, ImageIngestionError, VectorStoreError
from src.components.document_loader import DocumentLoader
from src.components.image_loader import ImageLoader
from src.components.embedding_engine import EmbeddingEngine

logger = get_logger(__name__)

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class IndexPipeline:
    

    def __init__(self):
        self.doc_loader   = DocumentLoader()
        self.image_loader = ImageLoader()
        self.engine       = EmbeddingEngine()
        logger.info("IndexPipeline ready")

    #Public API 

    def ingest_pdf(self, pdf_path: str) -> dict:
        
        pdf_path = Path(pdf_path).resolve()
        logger.info(f"IndexPipeline — ingesting PDF: {pdf_path.name}")

        try:
            docs = self.doc_loader.load(str(pdf_path))
        except DocumentLoadError as e:
            return {"source": pdf_path.name, "chunks_added": 0, "status": f"error: {e}"}

        try:
            self.engine.add_documents(docs)
        except VectorStoreError as e:
            return {"source": pdf_path.name, "chunks_added": 0, "status": f"error: {e}"}

        logger.info(f"PDF ingestion complete — {len(docs)} chunks stored")
        return {
            "source":       pdf_path.name,
            "chunks_added": len(docs),
            "status":       "success",
        }

    def ingest_image(self, image_path: str) -> dict:
        
        image_path = Path(image_path).resolve()
        logger.info(f"IndexPipeline — ingesting image: {image_path.name}")

        try:
            docs = self.image_loader.load(str(image_path))
        except ImageIngestionError as e:
            return {"source": image_path.name, "chunks_added": 0, "status": f"error: {e}"}

        try:
            self.engine.add_documents(docs)
        except VectorStoreError as e:
            return {"source": image_path.name, "chunks_added": 0, "status": f"error: {e}"}

        logger.info(f"Image ingestion complete — {len(docs)} chunks stored")
        return {
            "source":       image_path.name,
            "chunks_added": len(docs),
            "status":       "success",
        }

    def ingest_multiple_pdfs(self, pdf_paths: list[str]) -> list[dict]:
        return [self.ingest_pdf(p) for p in pdf_paths]

    def ingest_multiple_images(self, image_paths: list[str]) -> list[dict]:
        return [self.ingest_image(p) for p in image_paths]

    def delete_document(self, source_name: str) -> dict:
    
        logger.info(f"IndexPipeline — deleting: {source_name}")
        try:
            deleted = self.engine.delete_document(source_name)
        except VectorStoreError as e:
            return {"source": source_name, "chunks_deleted": 0, "status": f"error: {e}"}

        return {
            "source":          source_name,
            "chunks_deleted":  deleted,
            "status":          "success",
        }

    def list_documents(self) -> list[str]:
        """Return list of all ingested document names."""
        return self.engine.list_documents()

    def get_stats(self) -> dict:
        """Return vector store statistics."""
        docs = self.engine.list_documents()
        return {
            "total_chunks":    self.engine.count(),
            "total_documents": len(docs),
            "documents":       docs,
        }
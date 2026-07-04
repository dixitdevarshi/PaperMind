import sys
from pathlib import Path

from src.logger import get_logger
from src.exception import DocumentLoadError, ImageIngestionError, VectorStoreError
from src.components.document_loader import DocumentLoader
from src.components.image_loader import ImageLoader
from src.components.embedding_engine import EmbeddingEngine
from src.components.graph_builder import GraphBuilder

logger = get_logger(__name__)

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class IndexPipeline:
    """
    Orchestrates document ingestion end-to-end.
    Keeps both the vector store (ChromaDB) and the knowledge graph
    in sync with every new document — graph is updated incrementally,
    not rebuilt from scratch.

    Usage:
        pipeline = IndexPipeline()
        result   = pipeline.ingest_pdf("path/to/file.pdf")
        result   = pipeline.ingest_image("path/to/screenshot.png")
    """

    def __init__(self):
        self.doc_loader     = DocumentLoader()
        self.image_loader   = ImageLoader()
        self.engine         = EmbeddingEngine()
        self.graph_builder  = GraphBuilder()
        logger.info("IndexPipeline ready")

    def ingest_pdf(self, pdf_path: str) -> dict:
        """
        Ingest a single PDF file — stores chunks in ChromaDB and
        updates the knowledge graph incrementally.

        Returns dict with keys: source, chunks_added, status
        """
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

        # Keep knowledge graph in sync — incremental update, not full rebuild
        try:
            chunk_dicts = [
                {"text": d.page_content, "source": d.metadata["source"], "page": d.metadata["page"]}
                for d in docs
            ]
            self.graph_builder.add_chunks_to_existing_graph(chunk_dicts)
        except Exception as e:
            logger.warning(f"Graph update failed for '{pdf_path.name}': {e} — continuing without graph sync")

        logger.info(f"PDF ingestion complete — {len(docs)} chunks stored")
        return {
            "source":       pdf_path.name,
            "chunks_added": len(docs),
            "status":       "success",
        }

    def ingest_image(self, image_path: str) -> dict:
        """
        Ingest a single image/screenshot via Claude Vision — stores
        chunks in ChromaDB and updates the knowledge graph incrementally.

        Returns dict with keys: source, chunks_added, status
        """
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

        try:
            chunk_dicts = [
                {"text": d.page_content, "source": d.metadata["source"], "page": d.metadata["page"]}
                for d in docs
            ]
            self.graph_builder.add_chunks_to_existing_graph(chunk_dicts)
        except Exception as e:
            logger.warning(f"Graph update failed for '{image_path.name}': {e} — continuing without graph sync")

        logger.info(f"Image ingestion complete — {len(docs)} chunks stored")
        return {
            "source":       image_path.name,
            "chunks_added": len(docs),
            "status":       "success",
        }

    def ingest_multiple_pdfs(self, pdf_paths: list[str]) -> list[dict]:
        """Ingest multiple PDFs and return a result dict for each."""
        return [self.ingest_pdf(p) for p in pdf_paths]

    def ingest_multiple_images(self, image_paths: list[str]) -> list[dict]:
        """Ingest multiple images and return a result dict for each."""
        return [self.ingest_image(p) for p in image_paths]

    def delete_document(self, source_name: str) -> dict:
        """
        Remove all chunks for a document from the vector store.
        Note: does not currently remove the document's entities/edges
        from the knowledge graph — graph cleanup on delete is a known
        limitation, not handled in this version.

        Returns dict with keys: source, chunks_deleted, status
        """
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
import sys
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from src.logger import get_logger
from src.exception import EmbeddingError, VectorStoreError
from src.utils import get_vectorstore_path

logger = get_logger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "papermind_docs"
BATCH_SIZE      = 64


def _make_chunk_id(source: str, page: str | int, chunk_index: int, text: str) -> str:
    """
    Generate a deterministic ID for a chunk.
    Same chunk always gets the same ID — prevents duplicates on re-ingest.
    We hash: filename + page number + chunk index + first 80 chars of text.
    """
    raw = f"{source}__{page}__{chunk_index}__{text[:80]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class EmbeddingEngine:
    """
    Manages embeddings and the ChromaDB vector store.

    Usage:
        engine = EmbeddingEngine()
        engine.add_documents(docs)
        results = engine.query("What is GDPR?", n_results=5)
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}", sys)

        vectorstore_path = str(get_vectorstore_path())
        logger.info(f"Connecting to ChromaDB at: {vectorstore_path}")
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=vectorstore_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to ChromaDB: {e}", sys)

        logger.info(
            f"EmbeddingEngine ready — "
            f"collection '{COLLECTION_NAME}' has {self.collection.count()} chunks"
        )

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed a list of LangChain Documents and store in ChromaDB.
        Uses deterministic IDs so re-ingesting the same document
        updates existing chunks instead of creating duplicates.
        """
        if not documents:
            logger.warning("add_documents called with empty list — nothing to do")
            return

        logger.info(f"Embedding {len(documents)} chunks...")

        for batch_start in range(0, len(documents), BATCH_SIZE):
            batch = documents[batch_start: batch_start + BATCH_SIZE]

            texts     = [doc.page_content for doc in batch]
            metadatas = [doc.metadata     for doc in batch]

            # Deterministic IDs — same chunk = same ID every time
            ids = [
                _make_chunk_id(
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page", 0),
                    chunk_index=doc.metadata.get("chunk_index", i),
                    text=doc.page_content,
                )
                for i, doc in enumerate(batch)
            ]

            try:
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).tolist()
            except Exception as e:
                raise EmbeddingError(f"Embedding generation failed: {e}", sys)

            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
            except Exception as e:
                raise VectorStoreError(f"ChromaDB upsert failed: {e}", sys)

            logger.info(f"  Stored batch {batch_start // BATCH_SIZE + 1} ({len(batch)} chunks)")

        logger.info(f"Done — collection now has {self.collection.count()} total chunks")

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Semantic search — embed the query and return top-n matching chunks.
        Returns list of dicts with keys: text, metadata, score
        """
        if not query_text.strip():
            raise VectorStoreError("Query text cannot be empty", sys)

        try:
            query_embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True,
            ).tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}", sys)

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB query failed: {e}", sys)

        # ChromaDB returns distance (lower = more similar for cosine)
        # Convert to similarity score: score = 1 - distance
        chunks = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":     text,
                "metadata": metadata,
                "score":    round(1 - distance, 4),
            })

        logger.info(
            f"Query returned {len(chunks)} chunks "
            f"(top score: {chunks[0]['score'] if chunks else 'N/A'})"
        )
        return chunks

    def delete_document(self, source_name: str) -> int:
        """
        Remove all chunks belonging to a document by its source filename.
        Returns number of chunks deleted.
        """
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=["documents"],
            )
            ids_to_delete = results["ids"]

            if not ids_to_delete:
                logger.warning(f"No chunks found for source: {source_name}")
                return 0

            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for '{source_name}'")
            return len(ids_to_delete)

        except Exception as e:
            raise VectorStoreError(f"Delete failed for '{source_name}': {e}", sys)

    def list_documents(self) -> list[str]:
        """Return list of unique source document names in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            sources = list({m["source"] for m in results["metadatas"]})
            return sorted(sources)
        except Exception as e:
            raise VectorStoreError(f"Failed to list documents: {e}", sys)

    def count(self) -> int:
        return self.collection.count()
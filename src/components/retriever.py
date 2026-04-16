import sys
from dataclasses import dataclass

from src.logger import get_logger
from src.exception import RetrievalError
from src.components.embedding_engine import EmbeddingEngine

logger = get_logger(__name__)

#Confidence thresholds
HIGH_THRESHOLD   = 0.75
MEDIUM_THRESHOLD = 0.50

#Data class

@dataclass
class RetrievedChunk:
    """Single retrieved chunk with text, metadata, score, and confidence label."""
    text:       str
    source:     str
    page:       str | int
    section:    str
    score:      float
    confidence: str   # "High", "Medium", "Low"
    doc_type:   str


#Main class

class Retriever:
    

    def __init__(self, engine: EmbeddingEngine):
        self.engine = engine
        logger.info("Retriever ready")

    def retrieve(self, query: str, n_results: int = 5) -> list[RetrievedChunk]:
        
        if not query.strip():
            raise RetrievalError("Query cannot be empty", sys)

        if self.engine.count() == 0:
            raise RetrievalError(
                "Vector store is empty — ingest at least one document first", sys
            )

        logger.info(f"Retrieving top {n_results} chunks for query: '{query[:80]}'")

        raw_results = self.engine.query(query, n_results=n_results)

        chunks = []
        for result in raw_results:
            score      = result["score"]
            metadata   = result["metadata"]
            confidence = self._assign_confidence(score)

            chunks.append(RetrievedChunk(
                text       = result["text"],
                source     = metadata.get("source",  "Unknown"),
                page       = metadata.get("page",    "?"),
                section    = metadata.get("section", "Unknown"),
                score      = score,
                confidence = confidence,
                doc_type   = metadata.get("doc_type", "pdf"),
            ))

        # Sort by score descending (ChromaDB usually returns sorted, but be safe)
        chunks.sort(key=lambda c: c.score, reverse=True)

        logger.info(
            f"Retrieved {len(chunks)} chunks — "
            f"top score: {chunks[0].score:.4f} ({chunks[0].confidence}) "
            f"from '{chunks[0].source}' page {chunks[0].page}"
        )
        return chunks

    def retrieve_around_selection(
        self,
        selected_text: str,
        source_name:   str,
        n_results:     int = 5,
    ) -> list[RetrievedChunk]:
        
        if not selected_text.strip():
            raise RetrievalError("Selected text cannot be empty", sys)

        logger.info(
            f"Selection retrieval — source: '{source_name}', "
            f"selection: '{selected_text[:60]}...'"
        )

        # Retrieve broadly then filter by source
        raw_results = self.engine.query(selected_text, n_results=n_results * 3)

        filtered = [
            r for r in raw_results
            if r["metadata"].get("source") == source_name
        ]

        # Fall back to unfiltered if no results from that source
        if not filtered:
            logger.warning(
                f"No chunks found for source '{source_name}' — "
                f"falling back to global retrieval"
            )
            filtered = raw_results

        # Take top n_results
        filtered = filtered[:n_results]

        chunks = []
        for result in filtered:
            score    = result["score"]
            metadata = result["metadata"]
            chunks.append(RetrievedChunk(
                text       = result["text"],
                source     = metadata.get("source",  "Unknown"),
                page       = metadata.get("page",    "?"),
                section    = metadata.get("section", "Unknown"),
                score      = score,
                confidence = self._assign_confidence(score),
                doc_type   = metadata.get("doc_type", "pdf"),
            ))

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    #Internal

    @staticmethod
    def _assign_confidence(score: float) -> str:
        """Map cosine similarity score to High / Medium / Low label."""
        if score >= HIGH_THRESHOLD:
            return "High"
        elif score >= MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "Low"
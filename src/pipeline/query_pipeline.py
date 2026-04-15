"""
query_pipeline.py
─────────────────
Full query pipeline for PaperMind.

Flow:
  query → Retriever → ToolRouter → AnswerGenerator → response
"""

import sys
import re
from typing import Generator

from src.logger import get_logger
from src.exception import RetrievalError, RoutingError, LLMError
from src.components.embedding_engine import EmbeddingEngine
from src.components.retriever import Retriever, RetrievedChunk
from src.components.answer_generator import AnswerGenerator

logger = get_logger(__name__)

# ── Tool routing keywords ────────────────────────────────────────────────────

SUMMARIZE_KEYWORDS  = ["summarize", "summary", "overview", "zusammenfassung", "résumé"]
COMPARE_KEYWORDS    = ["compare", "difference", "versus", "vs", "vergleich", "comparer"]
EXTRACT_KEYWORDS    = ["extract", "list all", "find all", "enumerate", "aufzählen"]


class QueryPipeline:
    """
    Orchestrates retrieval → routing → generation end-to-end.

    Usage
    -----
    pipeline = QueryPipeline()
    result   = pipeline.query("What is GDPR Article 5?")
    result   = pipeline.query_selection("selected text", "GDPR_EN.pdf", "what does this mean?")
    """

    def __init__(self):
        self.engine    = EmbeddingEngine()
        self.retriever = Retriever(self.engine)
        self.generator = AnswerGenerator()
        logger.info("QueryPipeline ready")

    # ── Public API ───────────────────────────────────────────

    def query(self, question: str, n_results: int = 5) -> dict:
        """
        Full query: retrieve → route → generate → return response.

        Returns
        -------
        dict with keys:
            answer      : str
            sources     : list[dict]
            confidence  : str
            tool_used   : str
            model_used  : str
        """
        logger.info(f"QueryPipeline — question: '{question[:80]}'")

        # Step 1 — retrieve
        chunks = self._retrieve(question, n_results)

        # Step 2 — route
        tool = self._route(question)
        logger.info(f"Tool selected: {tool}")

        # Step 3 — generate
        response = self._generate(question, chunks, tool)
        response["tool_used"] = tool

        return response

    def query_stream(
        self,
        question:  str,
        n_results: int = 5,
    ) -> Generator[str, None, None]:
        """
        Streaming version of query().
        Yields answer tokens one by one.
        Retrieval and routing happen before streaming starts.
        """
        logger.info(f"QueryPipeline stream — question: '{question[:80]}'")

        chunks = self._retrieve(question, n_results)
        tool   = self._route(question)
        logger.info(f"Tool selected: {tool}")

        yield from self.generator.generate_stream(question, chunks)

    def query_selection(
        self,
        selected_text: str,
        source_name:   str,
        question:      str,
        n_results:     int = 5,
    ) -> dict:
        """
        Answer a question about a specific text selection from the PDF viewer.

        Parameters
        ----------
        selected_text : str — text the user highlighted
        source_name   : str — document filename being viewed
        question      : str — user's question about the selection
        """
        logger.info(
            f"Selection query — source: '{source_name}', "
            f"question: '{question[:60]}'"
        )

        # Combine selected text + question as the retrieval query
        combined_query = f"{selected_text}\n\n{question}"

        chunks = self.retriever.retrieve_around_selection(
            selected_text=selected_text,
            source_name=source_name,
            n_results=n_results,
        )

        if not chunks:
            return {
                "answer":     "Could not find relevant content for the selected text.",
                "sources":    [],
                "confidence": "Low",
                "tool_used":  "retriever",
                "model_used": "",
            }

        # Prepend the selected text as additional context
        augmented_question = (
            f"The user selected this text from the document:\n"
            f'"{selected_text}"\n\n'
            f"Their question: {question}"
        )

        response = self.generator.generate(augmented_question, chunks)
        response["tool_used"] = "retriever"
        return response

    def clear_memory(self) -> None:
        """Reset conversation memory."""
        self.generator.clear_memory()
        logger.info("QueryPipeline memory cleared")

    # ── Internal ─────────────────────────────────────────────

    def _retrieve(self, question: str, n_results: int) -> list[RetrievedChunk]:
        """Retrieve chunks, handle empty store gracefully."""
        try:
            return self.retriever.retrieve(question, n_results=n_results)
        except RetrievalError as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

    def _route(self, question: str) -> str:
        """
        Rule-based tool router.
        Checks question for keywords to select the right tool.

        Tools:
          summarizer  — summarization requests
          comparator  — comparison requests
          extractor   — structured extraction requests
          retriever   — default factual Q&A
        """
        q = question.lower()

        if any(kw in q for kw in SUMMARIZE_KEYWORDS):
            return "summarizer"
        if any(kw in q for kw in COMPARE_KEYWORDS):
            return "comparator"
        if any(kw in q for kw in EXTRACT_KEYWORDS):
            return "extractor"
        return "retriever"

    def _generate(
        self,
        question: str,
        chunks:   list[RetrievedChunk],
        tool:     str,
    ) -> dict:
        """Generate answer, adjusting the prompt style based on tool."""
        if tool == "summarizer":
            question = f"Please provide a concise summary based on the context. Original request: {question}"
        elif tool == "comparator":
            question = f"Please compare and contrast the relevant information across documents. Original request: {question}"
        elif tool == "extractor":
            question = f"Please extract and list the relevant structured information. Original request: {question}"

        try:
            return self.generator.generate(question, chunks)
        except LLMError as e:
            return {
                "answer":     f"Answer generation failed: {e}",
                "sources":    [],
                "confidence": "Low",
                "model_used": "",
            }
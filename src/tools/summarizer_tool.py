"""
summarizer_tool.py
──────────────────
Summarization tool — triggered when user asks for a summary or overview.
"""

from src.logger import get_logger
from src.components.retriever import Retriever
from src.components.answer_generator import AnswerGenerator

logger = get_logger(__name__)

SUMMARIZE_INSTRUCTION = (
    "Please provide a clear, structured summary of the following document content. "
    "Organize by main topics. Be concise. Cite page numbers where relevant."
)


class SummarizerTool:
    """
    Summarization tool.
    Retrieves the most representative chunks and generates a summary.

    Usage
    -----
    tool   = SummarizerTool(retriever, generator)
    result = tool.run("Summarize the Bosch supplier manual")
    """

    name        = "summarizer"
    description = "Summarizes document content when the user asks for an overview or summary."

    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator
        logger.info("SummarizerTool ready")

    def run(self, query: str, n_results: int = 8) -> dict:
        """
        Retrieve more chunks than usual (8) to give Claude
        a broader view for summarization.

        Returns
        -------
        dict with keys: answer, sources, confidence, tool_used, model_used
        """
        logger.info(f"SummarizerTool — query: '{query[:80]}'")

        chunks = self.retriever.retrieve(query, n_results=n_results)
        augmented = f"{SUMMARIZE_INSTRUCTION}\n\nOriginal request: {query}"
        result    = self.generator.generate(augmented, chunks)
        result["tool_used"] = self.name
        return result
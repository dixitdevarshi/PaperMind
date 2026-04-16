"""
comparator_tool.py
──────────────────
Comparison tool — triggered when user asks to compare across documents.
"""

from src.logger import get_logger
from src.components.retriever import Retriever
from src.components.answer_generator import AnswerGenerator

logger = get_logger(__name__)

COMPARE_INSTRUCTION = (
    "Please compare and contrast the relevant information from the documents. "
    "Structure your response as: \n"
    "1. Key similarities\n"
    "2. Key differences\n"
    "3. Summary\n"
    "Clearly label which document each point comes from. Cite page numbers."
)


class ComparatorTool:
    """
    Cross-document comparison tool.

    Usage
    -----
    tool   = ComparatorTool(retriever, generator)
    result = tool.run("Compare GDPR English and German versions on data subject rights")
    """

    name        = "comparator"
    description = "Compares information across multiple documents."

    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator
        logger.info("ComparatorTool ready")

    def run(self, query: str, n_results: int = 8) -> dict:
        """
        Retrieve chunks from all documents and generate a structured comparison.

        Returns
        -------
        dict with keys: answer, sources, confidence, tool_used, model_used
        """
        logger.info(f"ComparatorTool — query: '{query[:80]}'")

        chunks    = self.retriever.retrieve(query, n_results=n_results)
        augmented = f"{COMPARE_INSTRUCTION}\n\nOriginal request: {query}"
        result    = self.generator.generate(augmented, chunks)
        result["tool_used"] = self.name
        return result
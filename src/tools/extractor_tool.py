"""
extractor_tool.py
─────────────────
Structured extraction tool — triggered when user asks to extract or list items.
"""

from src.logger import get_logger
from src.components.retriever import Retriever
from src.components.answer_generator import AnswerGenerator

logger = get_logger(__name__)

EXTRACT_INSTRUCTION = (
    "Please extract and list the requested information in a structured format. "
    "Use numbered lists or tables where appropriate. "
    "Include the source document and page number for each extracted item. "
    "Do not add information that is not present in the context."
)


class ExtractorTool:
    """
    Structured data extraction tool.

    Usage
    -----
    tool   = ExtractorTool(retriever, generator)
    result = tool.run("List all data subject rights mentioned in the GDPR")
    """

    name        = "extractor"
    description = "Extracts and lists structured information from documents."

    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator
        logger.info("ExtractorTool ready")

    def run(self, query: str, n_results: int = 8) -> dict:
        """
        Retrieve chunks and extract structured information.

        Returns
        -------
        dict with keys: answer, sources, confidence, tool_used, model_used
        """
        logger.info(f"ExtractorTool — query: '{query[:80]}'")

        chunks    = self.retriever.retrieve(query, n_results=n_results)
        augmented = f"{EXTRACT_INSTRUCTION}\n\nOriginal request: {query}"
        result    = self.generator.generate(augmented, chunks)
        result["tool_used"] = self.name
        return result
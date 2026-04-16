from src.logger import get_logger
from src.components.retriever import Retriever, RetrievedChunk
from src.components.answer_generator import AnswerGenerator

logger = get_logger(__name__)


class RetrieverTool:
    
    name        = "retriever"
    description = "Answers factual questions by retrieving relevant document chunks."

    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator
        logger.info("RetrieverTool ready")

    def run(self, query: str, n_results: int = 5) -> dict:
        """
        Retrieve chunks and generate a grounded answer.

        Returns
        -------
        dict with keys: answer, sources, confidence, tool_used, model_used
        """
        logger.info(f"RetrieverTool — query: '{query[:80]}'")

        chunks = self.retriever.retrieve(query, n_results=n_results)
        result = self.generator.generate(query, chunks)
        result["tool_used"] = self.name
        return result
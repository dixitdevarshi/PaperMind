import sys
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.logger import get_logger
from src.exception import EvaluationError
from src.pipeline.query_pipeline import QueryPipeline
from src.utils import save_json, load_json

logger = get_logger(__name__)

DEFAULT_QA_PATH     = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation_qa.json"
DEFAULT_RESULT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation_results.json"

EMBEDDING_MODEL         = "paraphrase-multilingual-MiniLM-L12-v2"
RELEVANCE_THRESHOLD     = 0.45
FAITHFULNESS_THRESHOLD  = 0.40
RECALL_THRESHOLD        = 0.45


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for faithfulness scoring."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


class PaperMindEvaluator:

    def __init__(self):
        self.pipeline = QueryPipeline()
        logger.info(f"Loading evaluation model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("PaperMindEvaluator ready")

    def evaluate(
        self,
        qa_pairs:     list[dict] | None = None,
        qa_path:      str | None        = None,
        save_results: bool              = True,
    ) -> dict:
        if qa_pairs is None:
            path = qa_path or str(DEFAULT_QA_PATH)
            logger.info(f"Loading QA pairs from: {path}")
            try:
                qa_pairs = load_json(path)
            except Exception as e:
                raise EvaluationError(f"Failed to load QA pairs: {e}", sys)

        if not qa_pairs:
            raise EvaluationError("No QA pairs provided", sys)

        logger.info(f"Evaluating {len(qa_pairs)} QA pairs...")

        relevancy_scores:  list[float] = []
        faithfulness_scores: list[float] = []
        precision_scores:  list[float] = []
        recall_scores:     list[float] = []

        for i, pair in enumerate(qa_pairs):
            question     = pair["question"]
            ground_truth = pair["ground_truth"]

            logger.info(f"  [{i+1}/{len(qa_pairs)}] {question[:60]}")

            try:
                # Get answer and retrieved chunks from pipeline
                response = self.pipeline.query(question, n_results=5)
                answer   = response["answer"]
                chunks   = self.pipeline.retriever.retrieve(question, n_results=5)
                contexts = [chunk.text for chunk in chunks]

            except Exception as e:
                logger.warning(f"  Skipping pair {i+1}: {e}")
                continue

            # Compute all 4 metrics
            relevancy_scores.append(
                self._answer_relevancy(question, answer)
            )
            faithfulness_scores.append(
                self._faithfulness(answer, contexts)
            )
            precision_scores.append(
                self._context_precision(question, contexts)
            )
            recall_scores.append(
                self._context_recall(ground_truth, contexts)
            )

        if not relevancy_scores:
            raise EvaluationError("All QA pairs failed — check pipeline and vector store", sys)

        scores = {
            "answer_relevancy":  round(float(np.mean(relevancy_scores)),   4),
            "faithfulness":      round(float(np.mean(faithfulness_scores)), 4),
            "context_precision": round(float(np.mean(precision_scores)),    4),
            "context_recall":    round(float(np.mean(recall_scores)),       4),
            "num_evaluated":     len(relevancy_scores),
        }

        logger.info(f"Evaluation complete: {scores}")

        if save_results:
            save_json(scores, str(DEFAULT_RESULT_PATH))
            logger.info(f"Results saved to: {DEFAULT_RESULT_PATH}")

        return scores

    # Metric implementations

    def _answer_relevancy(self, question: str, answer: str) -> float:
        
        q_emb = self.model.encode(question, convert_to_numpy=True)
        a_emb = self.model.encode(answer,   convert_to_numpy=True)
        return _cosine_similarity(q_emb, a_emb)

    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Fraction of answer sentences that are grounded in at least one
        retrieved context chunk. A grounded sentence has cosine similarity
        above FAITHFULNESS_THRESHOLD with at least one chunk.
        """
        sentences = _split_sentences(answer)
        if not sentences:
            return 0.0

        context_embeddings = self.model.encode(contexts, convert_to_numpy=True)
        grounded = 0

        for sentence in sentences:
            s_emb = self.model.encode(sentence, convert_to_numpy=True)
            sims  = [_cosine_similarity(s_emb, c_emb) for c_emb in context_embeddings]
            if max(sims) >= FAITHFULNESS_THRESHOLD:
                grounded += 1

        return grounded / len(sentences)

    def _context_precision(self, question: str, contexts: list[str]) -> float:
        
        if not contexts:
            return 0.0

        q_emb = self.model.encode(question, convert_to_numpy=True)
        context_embeddings = self.model.encode(contexts, convert_to_numpy=True)

        relevant = sum(
            1 for c_emb in context_embeddings
            if _cosine_similarity(q_emb, c_emb) >= RELEVANCE_THRESHOLD
        )
        return relevant / len(contexts)

    def _context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        
        gt_sentences = _split_sentences(ground_truth)
        if not gt_sentences or not contexts:
            return 0.0

        context_embeddings = self.model.encode(contexts, convert_to_numpy=True)
        covered = 0

        for sentence in gt_sentences:
            s_emb = self.model.encode(sentence, convert_to_numpy=True)
            sims  = [_cosine_similarity(s_emb, c_emb) for c_emb in context_embeddings]
            if max(sims) >= RECALL_THRESHOLD:
                covered += 1

        return covered / len(gt_sentences)


# Run as standalone script
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    evaluator = PaperMindEvaluator()
    scores    = evaluator.evaluate()

    print("\n" + "="*45)
    print("  PaperMind Evaluation Results")
    print("="*45)
    print(f"  Answer Relevancy   : {scores['answer_relevancy']:.4f}")
    print(f"  Faithfulness       : {scores['faithfulness']:.4f}")
    print(f"  Context Precision  : {scores['context_precision']:.4f}")
    print(f"  Context Recall     : {scores['context_recall']:.4f}")
    print(f"  Pairs evaluated    : {scores['num_evaluated']}")
    print("="*45)
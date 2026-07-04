import sys
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.logger import get_logger
from src.exception import EvaluationError
from src.pipeline.query_pipeline import QueryPipeline
from src.components.graph_retriever import GraphRetriever
from src.utils import save_json, load_json

logger = get_logger(__name__)

SINGLE_FACT_QA_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation_qa.json"
MULTI_HOP_QA_PATH   = Path(__file__).resolve().parents[2] / "artifacts" / "multihop_qa.json"
RESULT_PATH         = Path(__file__).resolve().parents[2] / "artifacts" / "comparison_results.json"

EMBEDDING_MODEL        = "paraphrase-multilingual-MiniLM-L12-v2"
FAITHFULNESS_THRESHOLD = 0.40
RELEVANCE_THRESHOLD    = 0.45


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


class ComparisonEvaluator:
    """
    Runs the same faithfulness and context precision metrics on both
    vector retrieval and graph retrieval, separately for single-fact
    and multi-hop question sets.

    Usage:
        comparator = ComparisonEvaluator()
        results    = comparator.run_comparison()
    """

    def __init__(self):
        self.query_pipeline  = QueryPipeline()
        self.graph_retriever = GraphRetriever()
        self.model           = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("ComparisonEvaluator ready")

    def run_comparison(self, save_results: bool = True) -> dict:
        """
        Runs the full comparison across both question types and both
        retrieval methods. Returns a nested dict of scores.
        """
        try:
            single_fact_qa = load_json(str(SINGLE_FACT_QA_PATH))
        except Exception as e:
            raise EvaluationError(f"Failed to load single-fact QA: {e}", sys)

        try:
            multi_hop_qa = load_json(str(MULTI_HOP_QA_PATH))
        except Exception as e:
            raise EvaluationError(f"Failed to load multi-hop QA: {e}", sys)

        logger.info(
            f"Running comparison: {len(single_fact_qa)} single-fact questions, "
            f"{len(multi_hop_qa)} multi-hop questions"
        )

        results = {
            "single_fact": {
                "vector": self._evaluate_set(single_fact_qa, method="vector"),
                "graph":  self._evaluate_set(single_fact_qa, method="graph"),
            },
            "multi_hop": {
                "vector": self._evaluate_set(multi_hop_qa, method="vector"),
                "graph":  self._evaluate_set(multi_hop_qa, method="graph"),
            },
        }

        logger.info(f"Comparison complete: {results}")

        if save_results:
            save_json(results, str(RESULT_PATH))
            logger.info(f"Results saved to: {RESULT_PATH}")

        return results

    # ── Per-method evaluation ───────────────────────────────

    def _evaluate_set(self, qa_pairs: list[dict], method: str) -> dict:
        """
        Run one question set through one retrieval method.
        method is either "vector" or "graph".
        """
        faithfulness_scores = []
        precision_scores    = []
        zero_result_count   = 0

        for pair in qa_pairs:
            question     = pair["question"]
            ground_truth = pair["ground_truth"]

            contexts = self._retrieve_contexts(question, method)

            if not contexts:
                zero_result_count += 1
                faithfulness_scores.append(0.0)
                precision_scores.append(0.0)
                continue

            answer = self._generate_answer(question, contexts)

            faithfulness_scores.append(self._faithfulness(answer, contexts))
            precision_scores.append(self._context_precision(question, contexts))

        return {
            "faithfulness":      round(float(np.mean(faithfulness_scores)), 4),
            "context_precision": round(float(np.mean(precision_scores)),    4),
            "zero_result_rate":  round(zero_result_count / len(qa_pairs),   4),
            "num_questions":     len(qa_pairs),
        }

    def _retrieve_contexts(self, question: str, method: str) -> list[str]:
        """Retrieve context chunks using either vector or graph retrieval."""
        if method == "vector":
            chunks = self.query_pipeline.retriever.retrieve(question, n_results=5)
            return [c.text for c in chunks]

        elif method == "graph":
            chunks = self.graph_retriever.retrieve(question, hops=2, n_results=5)
            return [c.text for c in chunks]

        raise ValueError(f"Unknown method: {method}")

    def _generate_answer(self, question: str, contexts: list[str]) -> str:
        """
        Generate an answer from the given contexts using the same
        AnswerGenerator already used by the main pipeline — keeps the
        generation step identical so only retrieval differs.
        """
        from src.components.retriever import RetrievedChunk

        # Wrap raw context strings as RetrievedChunk objects so
        # AnswerGenerator can consume them the same way it always does
        fake_chunks = [
            RetrievedChunk(
                text=ctx, source="comparison", page="?", section="?",
                score=1.0, confidence="High", doc_type="comparison"
            )
            for ctx in contexts
        ]

        response = self.query_pipeline.generator.generate(question, fake_chunks)
        return response["answer"]

    # ── Metrics (same logic as evaluator.py) ────────────────

    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
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


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    comparator = ComparisonEvaluator()
    results    = comparator.run_comparison()

    print("\n" + "=" * 60)
    print("  PaperMind — Vector vs Graph Retrieval Comparison")
    print("=" * 60)

    for question_type in ["single_fact", "multi_hop"]:
        print(f"\n  {question_type.upper().replace('_', ' ')}")
        print(f"  {'-' * 56}")
        for method in ["vector", "graph"]:
            scores = results[question_type][method]
            print(f"  {method.capitalize():<8} | "
                  f"Faithfulness: {scores['faithfulness']:.4f} | "
                  f"Precision: {scores['context_precision']:.4f} | "
                  f"Zero-result rate: {scores['zero_result_rate']:.4f}")

    print("\n" + "=" * 60)
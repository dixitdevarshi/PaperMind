"""
ragas_evaluator.py
RAGAS evaluation pipeline for PaperMind.
Runs as a standalone script: python -m src.evaluation.ragas_evaluator
"""

import sys
import os
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.logger import get_logger
from src.exception import EvaluationError
from src.pipeline.query_pipeline import QueryPipeline
from src.utils import save_json, load_json

logger = get_logger(__name__)

DEFAULT_QA_PATH    = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation_qa.json"
DEFAULT_RESULT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation_results.json"


class RagasEvaluator:
    """
    Runs RAGAS evaluation using OpenAI for metric computation.
    Your pipeline still uses Claude — OpenAI is only used by RAGAS
    internally to score faithfulness and relevancy.

    Requires OPENAI_API_KEY in .env for RAGAS metrics.
    Your ANTHROPIC_API_KEY is still used for all answer generation.

    Usage:
        evaluator = RagasEvaluator()
        scores    = evaluator.evaluate()
    """

    def __init__(self):
        self.pipeline = QueryPipeline()
        logger.info("RagasEvaluator ready")

    def evaluate(
        self,
        qa_pairs:     list[dict] | None = None,
        qa_path:      str | None        = None,
        save_results: bool              = True,
    ) -> dict:
        """
        Run RAGAS evaluation.

        Parameters:
            qa_pairs     : list of {question, ground_truth} dicts
            qa_path      : path to JSON file with QA pairs
            save_results : save scores to artifacts/evaluation_results.json
        """
        if qa_pairs is None:
            path = qa_path or str(DEFAULT_QA_PATH)
            logger.info(f"Loading QA pairs from: {path}")
            try:
                qa_pairs = load_json(path)
            except Exception as e:
                raise EvaluationError(f"Failed to load QA pairs: {e}", sys)

        if not qa_pairs:
            raise EvaluationError("No QA pairs provided for evaluation", sys)

        logger.info(f"Running RAGAS on {len(qa_pairs)} QA pairs...")

        try:
            dataset = self._build_dataset(qa_pairs)
        except Exception as e:
            raise EvaluationError(f"Failed to build evaluation dataset: {e}", sys)

        try:
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
        except Exception as e:
            raise EvaluationError(f"RAGAS evaluation failed: {e}", sys)

        scores = {
            "faithfulness":      round(float(result["faithfulness"]),      4),
            "answer_relevancy":  round(float(result["answer_relevancy"]),  4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall":    round(float(result["context_recall"]),    4),
        }

        logger.info(f"RAGAS scores: {scores}")

        if save_results:
            save_json(scores, str(DEFAULT_RESULT_PATH))
            logger.info(f"Results saved to: {DEFAULT_RESULT_PATH}")

        return scores

    def _build_dataset(self, qa_pairs: list[dict]) -> Dataset:
        """
        Run each question through the pipeline and collect
        question, answer, contexts, ground_truth for RAGAS.
        """
        questions:    list[str]       = []
        answers:      list[str]       = []
        contexts:     list[list[str]] = []
        ground_truth: list[str]       = []

        for i, pair in enumerate(qa_pairs):
            q  = pair["question"]
            gt = pair["ground_truth"]

            logger.info(f"  [{i+1}/{len(qa_pairs)}] '{q[:60]}'")

            try:
                response = self.pipeline.query(q, n_results=5)
                answer   = response["answer"]
                chunks   = self.pipeline.retriever.retrieve(q, n_results=5)
                ctx      = [chunk.text for chunk in chunks]
            except Exception as e:
                logger.warning(f"  Skipping pair {i+1}: {e}")
                continue

            questions.append(q)
            answers.append(answer)
            contexts.append(ctx)
            ground_truth.append(gt)

        if not questions:
            raise EvaluationError(
                "All QA pairs failed — check your pipeline and vector store", sys
            )

        return Dataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    evaluator = RagasEvaluator()
    scores    = evaluator.evaluate()
    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in scores.items():
        print(f"  {metric:<25} {score:.4f}")
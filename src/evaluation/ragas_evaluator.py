import sys
from pathlib import Path
from typing import Any

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

#Constants

METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

DEFAULT_QA_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "evaluation_qa.json"
)


class RagasEvaluator:

    def __init__(self):
        self.pipeline = QueryPipeline()
        logger.info("RagasEvaluator ready")

    #Public API

    def evaluate(
        self,
        qa_pairs:    list[dict] | None = None,
        qa_path:     str | None        = None,
        save_results: bool             = True,
    ) -> dict:
        
        # Load QA pairs
        if qa_pairs is None:
            path = qa_path or str(DEFAULT_QA_PATH)
            logger.info(f"Loading QA pairs from: {path}")
            try:
                qa_pairs = load_json(path)
            except Exception as e:
                raise EvaluationError(f"Failed to load QA pairs: {e}", sys)

        if not qa_pairs:
            raise EvaluationError("No QA pairs provided for evaluation", sys)

        logger.info(f"Running RAGAS evaluation on {len(qa_pairs)} QA pairs...")

        # Build RAGAS dataset
        try:
            dataset = self._build_dataset(qa_pairs)
        except Exception as e:
            raise EvaluationError(f"Failed to build evaluation dataset: {e}", sys)

        # Run evaluation
        try:
            result = evaluate(dataset, metrics=METRICS)
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
            output_path = str(
                Path(__file__).resolve().parents[2]
                / "artifacts"
                / "evaluation_results.json"
            )
            save_json(scores, output_path)
            logger.info(f"Results saved to: {output_path}")

        return scores

    #Internal
    def _build_dataset(self, qa_pairs: list[dict]) -> Dataset:
        
        questions:   list[str]       = []
        answers:     list[str]       = []
        contexts:    list[list[str]] = []
        ground_truth: list[str]      = []

        for i, pair in enumerate(qa_pairs):
            q  = pair["question"]
            gt = pair["ground_truth"]

            logger.info(f"  Evaluating QA pair {i+1}/{len(qa_pairs)}: '{q[:60]}'")

            try:
                # Get answer + retrieved chunks from pipeline
                response = self.pipeline.query(q, n_results=5)
                answer   = response["answer"]
                sources  = response["sources"]

                # Re-retrieve chunks to get raw text for context
                chunks = self.pipeline.retriever.retrieve(q, n_results=5)
                ctx    = [chunk.text for chunk in chunks]

            except Exception as e:
                logger.warning(f"  Skipping pair {i+1} due to error: {e}")
                continue

            questions.append(q)
            answers.append(answer)
            contexts.append(ctx)
            ground_truth.append(gt)

        if not questions:
            raise EvaluationError(
                "All QA pairs failed during evaluation — check your pipeline", sys
            )

        return Dataset.from_dict({
            "question":    questions,
            "answer":      answers,
            "contexts":    contexts,
            "ground_truth": ground_truth,
        })
"""
evals/eval_rag_pipeline.py
==========================
End-to-end evaluation of the full RAG pipeline (retrieve → rerank → generate).

Runs the RAG Triad metrics:
  - ContextualRelevancy: are the retrieved chunks relevant to the query?
  - Faithfulness: is the generated answer grounded in the context?
  - AnswerRelevancy: does the answer actually address the query?

    python -m evals.eval_rag_pipeline
"""
# eval_rag_pipeline.py
import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from src.gemini_model import gemini_judge
from evals.harness import load_goldens, summarize_by_metric, print_summary

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
JUDGE_MODEL = gemini_judge
THRESHOLD = 0.7


def run(rag):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
    ]

    # 4. EVALUATE
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
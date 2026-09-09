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

import os
import sys
import json
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

# Prevent all HuggingFace download attempts (Mixtral tokenizer fallback is harmless)
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

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
JUDGE_MODEL = gemini_judge     # use Gemini 3.5 Flash Lite via our DeepEval wrapper
THRESHOLD = 0.7


# 1. LOAD queries (we only need the queries — context comes from the pipeline now)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["query"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],       # what the generator produced
            retrieval_context=result["context"],  # what the RETRIEVER returned
        )
    )


# 3. THE THREE TRIAD METRICS — sequential to respect Mistral rate limits
metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
]


# 4. EVALUATE — sequential to avoid Mistral rate limit timeouts
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
    cache_config=CacheConfig(write_cache=False, use_cache=False),
)
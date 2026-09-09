"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault — the context was already correct.

    python -m evals.eval_generator
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
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.gemini_model import gemini_judge

from src.generator import generate   # your generator: generate(query, context) -> answer

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = gemini_judge
THRESHOLD = 0.7


# 1. LOAD the faithfulness golden set (query + ideal_context)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
test_cases = []
for g in goldens:
    context = g["ideal_context"]              # known-good context (list of chunk strings)
    answer = generate(g["query"], context)    # RUN the generator -> actual_output

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,             # the generated answer we're judging
            retrieval_context=context,        # faithfulness checks the answer against THIS
            # no expected_output — faithfulness never reads it
        )
    )


# 3. THE METRIC — decomposes actual_output into claims, attributes each to context
metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,   # prints WHY each score — shows which claims were unsupported
        async_mode=False,
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=False,
    ),
]


# 4. EVALUATE — runs the metric on every case sequentially (avoids Mistral rate limits)
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
    cache_config=CacheConfig(write_cache=False, use_cache=False),
)
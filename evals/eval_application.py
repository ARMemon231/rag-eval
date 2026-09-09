"""
evals/eval_application.py
=========================
Application-level evaluation of the full RAG pipeline.

    python -m evals.eval_application
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
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline
from src.gemini_model import gemini_judge

GOLDEN_PATH = "goldens/correctness_goldens.json"
JUDGE_MODEL = gemini_judge     # use Gemini 3.5 Flash Lite via our DeepEval wrapper
THRESHOLD = 0.7


# 1. LOAD queries + ideal answers
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["question"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],
            expected_output=g["ideal_answer"],
        )
    )


# 3. THREE APPLICATION-LEVEL QUALITY METRICS

# 3a. CORRECTNESS — reference-based, judges TRUTH (not coverage or length)
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
        "Do NOT deduct for brevity, missing elaboration, or omitted points — omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
        Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
        Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
    async_mode=False,
)

# 3b. COMPLETENESS — reference-based, judges COVERAGE (not correctness)
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly — factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
        Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
    async_mode=False,
)

# 3c. STYLE — reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
        "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
        "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
        "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
        "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
        "Do NOT reward or penalize based on correctness, completeness, or length — only on style and tone.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes."),
        Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
        Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
        Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
    async_mode=False,
)




# 4. EVALUATE — sequential to avoid Mistral rate limit timeouts
evaluate(
    test_cases=test_cases,
    metrics=[correctness, completeness, style],
    async_config=AsyncConfig(run_async=False),
    cache_config=CacheConfig(write_cache=False, use_cache=False),
)
# evaluate(test_cases=test_cases, metrics=[correctness, completeness, style], async_config=AsyncConfig(run_async=False))

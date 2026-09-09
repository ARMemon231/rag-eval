"""
Shared Gemini model wrapper for DeepEval and RAG components.

Wraps ChatGoogleGenerativeAI from langchain_google_genai so DeepEval
and custom pipeline components can use Gemini (gemini-3.5-flash-lite)
without hitting Mistral rate limits.

Usage:
    from src.gemini_model import gemini_judge
    ContextualRecallMetric(model=gemini_judge, ...)
"""

import os
import re
import json
import asyncio
import time
import sys
from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _extract_text(content) -> str:
    """Extract string text from message content or parsed output."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)
    return str(content)


class GeminiDeepEvalModel(DeepEvalBaseLLM):
    """Wraps ChatGoogleGenerativeAI so DeepEval can use Gemini models."""

    MAX_RETRIES = 5
    BASE_DELAY = 2  # seconds
    _semaphore = None

    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self._client = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key,
        )

    @property
    def semaphore(self):
        if GeminiDeepEvalModel._semaphore is None:
            GeminiDeepEvalModel._semaphore = asyncio.Semaphore(5)
        return GeminiDeepEvalModel._semaphore

    def load_model(self):
        return self._client

    def _parse_response(self, text: str, schema):
        """Extract the first valid JSON object matching schema from response."""
        # 1) Try markdown fences first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                data = json.loads(fence_match.group(1))
                return schema(**data)
            except (json.JSONDecodeError, Exception):
                pass

        # 2) Fallback: brace counting
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate)
                        return schema(**data)
                    except (json.JSONDecodeError, Exception):
                        break

        # 3) Stripping markdown
        cleaned = re.sub(r"```(?:json)?|```|<[^>]+>", "", text).strip()
        if cleaned:
            try:
                data = json.loads(cleaned)
                return schema(**data)
            except (json.JSONDecodeError, Exception):
                pass

        # 4) Single-field schema wrapper fallback
        fields = schema.model_fields
        if len(fields) == 1:
            field_name = list(fields.keys())[0]
            clean_text = re.sub(r"```(?:json)?|```|<[^>]+>", "", text).strip()
            return schema(**{field_name: clean_text})

        raise ValueError(
            f"Could not parse Gemini response into {schema.__name__}: {text[:300]}"
        )

    def _add_json_hint(self, prompt: str, schema) -> str:
        fields = {
            k: v.annotation.__name__ if hasattr(v.annotation, "__name__") else str(v.annotation)
            for k, v in schema.model_fields.items()
        }
        return (
            prompt
            + f"\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object matching "
            + f"this exact schema: {json.dumps(fields)}. "
            + "Do NOT include markdown fences, code blocks, explanations, or any other text."
        )

    def generate(self, prompt: str, schema=None):
        """Synchronous generation."""
        actual_prompt = self._add_json_hint(prompt, schema) if schema else prompt
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.invoke([HumanMessage(content=actual_prompt)])
                text = _extract_text(response.content)
                if schema is not None:
                    return self._parse_response(text, schema)
                return text
            except Exception as e:
                is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
                if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                    wait = self.BASE_DELAY * (2 ** attempt)
                    print(f"  ⏳ Gemini rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    async def a_generate(self, prompt: str, schema=None):
        """Asynchronous generation."""
        actual_prompt = self._add_json_hint(prompt, schema) if schema else prompt
        async with self.semaphore:
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = await self._client.ainvoke([HumanMessage(content=actual_prompt)])
                    text = _extract_text(response.content)
                    if schema is not None:
                        return self._parse_response(text, schema)
                    return text
                except Exception as e:
                    is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
                    if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                        wait = self.BASE_DELAY * (2 ** attempt)
                        print(f"  ⏳ Gemini rate limited, retrying in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        raise

    def get_model_name(self) -> str:
        return self.model_name


# Pre-built instance for DeepEval evals
gemini_judge = GeminiDeepEvalModel(model_name="gemini-3.5-flash-lite")

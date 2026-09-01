"""
Shared Mistral model wrapper for DeepEval.

DeepEval's metrics and synthesizers expect either an OpenAI key or a
`DeepEvalBaseLLM` subclass.  This module provides `MistralDeepEvalModel`
which wraps `ChatMistralAI` from langchain_mistralai so every DeepEval
component can use Mistral without an OpenAI key.

Usage:
    from src.mistral_model import mistral_judge
    ContextualRecallMetric(model=mistral_judge, ...)
"""

import os, re, json, asyncio, time
from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_mistralai import ChatMistralAI

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

# Prevent all HuggingFace download attempts — the Mixtral tokenizer
# fallback (uses len()) is harmless, but the retry storm wastes 30+ seconds.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class MistralDeepEvalModel(DeepEvalBaseLLM):
    """Wraps langchain_mistralai's ChatMistralAI so DeepEval can use it
    without requiring an OpenAI API key."""

    MAX_RETRIES = 5
    BASE_DELAY = 3  # seconds
    _semaphore = None

    def __init__(self, model_name: str = "mistral-medium-latest"):
        self.model_name = model_name
        self._client = ChatMistralAI(
            model=model_name,
            mistral_api_key=os.environ["MISTRAL_API_KEY"],
        )

    @property
    def semaphore(self):
        if MistralDeepEvalModel._semaphore is None:
            MistralDeepEvalModel._semaphore = asyncio.Semaphore(1)
        return MistralDeepEvalModel._semaphore

    def load_model(self):
        return self._client

    def _parse_response(self, text: str, schema):
        """Extract the first valid JSON object from Mistral's response,
        regardless of markdown fences, HTML tags, or explanatory text."""
        # 1) Try to grab the content inside ```json ... ``` fences first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                data = json.loads(fence_match.group(1))
                return schema(**data)
            except (json.JSONDecodeError, Exception):
                pass

        # 2) Fallback: find the first balanced { ... } block by brace counting
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

        # 3) Try stripping markdown/HTML then parsing
        cleaned = re.sub(r"```(?:json)?|```|<[^>]+>", "", text).strip()
        if cleaned:
            try:
                data = json.loads(cleaned)
                return schema(**data)
            except (json.JSONDecodeError, Exception):
                pass

        # 4) Last resort: if schema has a single string field, wrap plain text
        fields = schema.model_fields
        if len(fields) == 1:
            field_name = list(fields.keys())[0]
            clean_text = re.sub(r"```(?:json)?|```|<[^>]+>", "", text).strip()
            return schema(**{field_name: clean_text})

        raise ValueError(
            f"Could not parse Mistral response into {schema.__name__}: {text[:300]}"
        )

    def _add_json_hint(self, prompt: str, schema) -> str:
        """Append a JSON formatting hint so Mistral returns structured output."""
        fields = {k: v.annotation.__name__ if hasattr(v.annotation, '__name__') else str(v.annotation)
                  for k, v in schema.model_fields.items()}
        return (
            prompt
            + f"\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object matching "
            + f"this exact schema: {json.dumps(fields)}. "
            + "Do NOT include markdown fences, code blocks, explanations, or any other text."
        )

    def generate(self, prompt: str, schema=None):
        """Sync call with retry on rate-limit errors."""
        from langchain_core.messages import HumanMessage
        import httpx
        actual_prompt = self._add_json_hint(prompt, schema) if schema else prompt
        for attempt in range(self.MAX_RETRIES):
            try:
                time.sleep(1.0)  # Throttling delay
                response = self._client.invoke([HumanMessage(content=actual_prompt)])
                if schema is not None:
                    return self._parse_response(response.content, schema)
                return response.content
            except (httpx.HTTPStatusError, Exception) as e:
                is_429 = getattr(getattr(e, 'response', None), 'status_code', None) == 429 or "429" in str(e)
                if is_429 and attempt < self.MAX_RETRIES - 1:
                    wait = self.BASE_DELAY * (2 ** attempt)
                    print(f"  ⏳ Rate limited, retrying in {wait}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    raise

    async def a_generate(self, prompt: str, schema=None):
        """Async call with semaphore + retry on rate-limit errors."""
        from langchain_core.messages import HumanMessage
        import httpx
        actual_prompt = self._add_json_hint(prompt, schema) if schema else prompt
        async with self.semaphore:
            for attempt in range(self.MAX_RETRIES):
                try:
                    await asyncio.sleep(1.0)  # Throttling delay
                    response = await self._client.ainvoke([HumanMessage(content=actual_prompt)])
                    if schema is not None:
                        return self._parse_response(response.content, schema)
                    return response.content
                except (httpx.HTTPStatusError, Exception) as e:
                    is_429 = getattr(getattr(e, 'response', None), 'status_code', None) == 429 or "429" in str(e)
                    if is_429 and attempt < self.MAX_RETRIES - 1:
                        wait = self.BASE_DELAY * (2 ** attempt)
                        print(f"  ⏳ Rate limited, retrying in {wait}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        await asyncio.sleep(wait)
                    else:
                        raise

    def get_model_name(self) -> str:
        return self.model_name


# Pre-built instance for convenience — import this directly
mistral_judge = MistralDeepEvalModel(model_name="mistral-medium-latest")

"""
Mistral-based reranker — no HuggingFace downloads needed.

Instead of a local CrossEncoder, this sends one Mistral API call per query
asking the model to score each candidate chunk's relevance (0-10).
"""

import re, json, time
from src.retriever import load_store
from src.mistral_model import MistralDeepEvalModel


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_store()
        self.model = MistralDeepEvalModel(model_name="mistral-medium-latest")
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking

    def _score_documents(self, query, docs):
        """Ask Mistral to score each document's relevance to the query.
        Returns a list of float scores (one per doc)."""
        # Build a numbered list of document excerpts (truncate to save tokens)
        doc_list = ""
        for i, doc in enumerate(docs):
            excerpt = doc.page_content[:500]  # first 500 chars is enough
            doc_list += f"\n[DOC {i}]: {excerpt}\n"

        prompt = f"""You are a relevance judge. Given a query and a list of documents, 
score each document's relevance to the query on a scale of 0 to 10.
- 10 = perfectly answers the query
- 0 = completely irrelevant

Query: {query}

Documents:{doc_list}

Respond with ONLY a JSON array of numbers (one score per document, in order).
Example for 3 documents: [8, 2, 6]
Do NOT include any other text, explanation, or markdown."""

        for attempt in range(3):
            try:
                time.sleep(1.0)  # throttle
                response = self.model.generate(prompt)

                # Extract the JSON array from the response
                # Try direct parse first
                cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
                match = re.search(r"\[[\d\s,\.]+\]", cleaned)
                if match:
                    scores = json.loads(match.group())
                    if len(scores) == len(docs):
                        return [float(s) for s in scores]

                # If wrong length, pad or truncate
                scores = json.loads(match.group()) if match else [5.0] * len(docs)
                while len(scores) < len(docs):
                    scores.append(0.0)
                return [float(s) for s in scores[: len(docs)]]

            except Exception as e:
                if attempt < 2:
                    wait = 3 * (2 ** attempt)
                    print(f"  ⏳ Reranker retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    # On final failure, return equal scores (no reranking)
                    print(f"  ⚠️ Reranker failed, falling back to original order")
                    return [1.0] * len(docs)

    def invoke(self, query):
        # 1. OVER-RETRIEVE: fast bi-encoder, deliberately more than we need
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. RERANK: score each chunk's relevance via Mistral
        scores = self._score_documents(query, candidates)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]
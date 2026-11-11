"""Thin wrapper around the OpenAI-compatible client (OpenAI, LiteLLM proxy or Groq)."""

import json
import re
from functools import lru_cache
from typing import Callable

from openai import OpenAI

from app import config
from app.local_embeddings import TfidfEmbedder


@lru_cache(maxsize=1)
def client() -> OpenAI:
    # max_retries: the SDK backs off and retries on 429 rate limits (useful on Groq's free tier).
    return OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL, max_retries=5)


def chat(messages: list[dict], **kwargs):
    """One chat completion call. Returns the assistant message object."""
    response = client().chat.completions.create(model=config.CHAT_MODEL, messages=messages, **kwargs)
    return response.choices[0].message


def chat_json(system: str, user: str) -> dict:
    """Ask the model for a JSON object and parse it (tolerates ```json fences)."""
    message = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    return parse_json(message.content or "")


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with the API in batches; returns one vector per text, in order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        response = client().embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


def get_embedder() -> Callable[[list[str]], list[list[float]]]:
    """API embeddings, or local TF-IDF (fitted on the corpus when the index is built)."""
    return TfidfEmbedder() if config.EMBEDDING_PROVIDER == "local" else embed

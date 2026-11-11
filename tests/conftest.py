"""Offline test helpers: a tiny fixture corpus and a deterministic fake embedding."""

import hashlib
import re
from pathlib import Path

import pytest

from app import corpus

FIXTURES = Path(__file__).parent / "fixtures"


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-words hashed into 128 dims: enough for word-overlap similarity, no API needed."""
    vectors = []
    for text in texts:
        vec = [0.0] * 128
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1.0
        vectors.append(vec)
    return vectors


@pytest.fixture
def index(monkeypatch):
    built = corpus.CorpusIndex.build(FIXTURES, embed_fn=fake_embed)
    monkeypatch.setattr(corpus, "_index", built)
    return built

"""Local TF-IDF vectors with numpy: in-memory similarity without any embeddings API.

Used with Groq (which has no embeddings endpoint) or when EMBEDDING_PROVIDER=local.
Words and word pairs (bigrams) are weighted by TF-IDF; vectors are L2-normalised,
so the corpus search stays a plain cosine similarity.
"""

import math
import re
from collections import Counter

import numpy as np

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this
to was were will with what which who how when should can may not""".split())


def tokenize(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class TfidfEmbedder:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float32)

    def fit(self, texts: list[str]) -> None:
        """Learn the vocabulary and IDF weights from the corpus chunks."""
        doc_freq = Counter(term for text in texts for term in set(tokenize(text)))
        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        n = len(texts)
        self.idf = np.array(
            [math.log((1 + n) / (1 + doc_freq[t])) + 1 for t in sorted(doc_freq)], dtype=np.float32
        )

    def __call__(self, texts: list[str]) -> list[list[float]]:
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(tokenize(text)).items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = (1 + math.log(count)) * self.idf[col]  # sublinear TF
        return matrix.tolist()

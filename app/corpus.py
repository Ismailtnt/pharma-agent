"""Load the markdown corpus, split it into section chunks, and search it in memory.

Chunks follow the markdown heading structure (one chunk per section, long
sections split on paragraphs), so a hit always maps back to a file + section
that can be cited. Similarity is plain cosine over normalised embeddings held
in a numpy matrix: ~20 files fit easily in memory, no vector DB needed.
"""

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app import config, llm

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    source: str   # file name, e.g. dailymed-warfarin.md
    title: str    # document title (first H1, or file stem)
    section: str  # heading path, e.g. "Contraindications"
    text: str

    def header(self) -> str:
        return f"[{self.source}] {self.title} > {self.section}"


def split_markdown(source: str, content: str, max_chars: int = config.CHUNK_MAX_CHARS) -> list[Chunk]:
    """Split one markdown document into chunks, one per (sub)section."""
    title = Path(source).stem
    headings: dict[int, str] = {}
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]

    for line in content.splitlines():
        match = HEADING.match(line.strip())
        if match:
            level, text = len(match.group(1)), match.group(2).strip()
            if level == 1 and title == Path(source).stem:
                title = text
                continue
            headings = {lvl: h for lvl, h in headings.items() if lvl < level}
            headings[level] = text
            sections.append((" > ".join(headings[lvl] for lvl in sorted(headings)), []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for section, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for piece in _split_long(body, max_chars):
            chunks.append(Chunk(source=source, title=title, section=section, text=piece))
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    """Group paragraphs so that each piece stays under max_chars."""
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for paragraph in re.split(r"\n\s*\n", text):
        if current and len(current) + len(paragraph) > max_chars:
            pieces.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


class CorpusIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embed_fn: Callable):
        self.chunks = chunks
        self.vectors = vectors
        self.embed_fn = embed_fn

    @classmethod
    def build(cls, corpus_dir: Path = config.CORPUS_DIR, embed_fn: Callable | None = None) -> "CorpusIndex":
        files = sorted(p for p in Path(corpus_dir).glob("*") if p.suffix.lower() in {".md", ".txt"})
        if not files:
            raise FileNotFoundError(f"No .md/.txt files found in {corpus_dir}")
        chunks = [c for f in files for c in split_markdown(f.name, f.read_text(encoding="utf-8"))]
        # Prefix each chunk with its file/section so the vector knows which drug it is about.
        texts = [f"{c.header()}\n{c.text}" for c in chunks]
        embed_fn = embed_fn or llm.get_embedder()
        if hasattr(embed_fn, "fit"):  # local TF-IDF learns its vocabulary from the corpus
            embed_fn.fit(texts)
        vectors = _normalise(np.array(embed_fn(texts), dtype=np.float32))
        return cls(chunks, vectors, embed_fn)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, k: int = config.TOP_K, source_filter: str | None = None) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity, optionally restricted to files whose name contains source_filter."""
        query_vec = _normalise(np.array(self.embed_fn([query]), dtype=np.float32))[0]
        scores = self.vectors @ query_vec
        candidates = range(len(self.chunks))
        if source_filter:
            needle = source_filter.lower().strip()
            candidates = [i for i in candidates if needle in self.chunks[i].source.lower()]
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]

    def sections_of(self, source: str) -> list[str]:
        return list(dict.fromkeys(c.section for c in self.chunks if c.source == source))


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


# --- Lazily built singleton -------------------------------------------------
_index: CorpusIndex | None = None
_lock = threading.Lock()


def get_index() -> CorpusIndex:
    global _index
    with _lock:
        if _index is None:
            _index = CorpusIndex.build()
        return _index


def index_ready() -> bool:
    return _index is not None

from app.corpus import CorpusIndex
from app.local_embeddings import TfidfEmbedder

from tests.conftest import FIXTURES


def test_tfidf_index_ranks_the_right_document_first():
    index = CorpusIndex.build(FIXTURES, embed_fn=TfidfEmbedder())
    chunk, score = index.search("apixaban hypersensitivity")[0]
    assert chunk.source == "dailymed-apixaban.md" and score > 0


def test_tfidf_finds_interaction_section():
    index = CorpusIndex.build(FIXTURES, embed_fn=TfidfEmbedder())
    chunk, _ = index.search("sertraline bleeding risk monitoring")[0]
    assert chunk.section == "Drug Interactions"

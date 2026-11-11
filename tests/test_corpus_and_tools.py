from app import tools
from app.corpus import split_markdown


def test_split_markdown_keeps_title_and_sections():
    chunks = split_markdown("x.md", "# Drug X\n\nintro\n\n## Dosage\n\n10 mg\n\n### Renal\n\nreduce dose")
    assert [c.section for c in chunks] == ["Introduction", "Dosage", "Dosage > Renal"]
    assert all(c.title == "Drug X" for c in chunks)


def test_search_can_filter_by_drug(index):
    hits = index.search("contraindications bleeding", source_filter="apixaban")
    assert hits and all(c.source == "dailymed-apixaban.md" for c, _ in hits)


def test_read_section_returns_full_section(index):
    text, chunks = tools.read_section(index, "warfarin", "contraindications")
    assert "Malignant hypertension" in text
    assert {c.source for c in chunks} == {"dailymed-warfarin.md"}


def test_read_section_lists_sections_when_missing(index):
    text, chunks = tools.read_section(index, "warfarin", "overdosage")
    assert chunks == [] and "Drug Interactions" in text


def test_glossary_lookup(index):
    text, _ = tools.lookup_glossary(index, "INR")
    assert "International Normalized Ratio" in text

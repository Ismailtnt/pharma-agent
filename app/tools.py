"""Local tools the agent can call. The LLM decides which one to use (OpenAI function calling).

Every tool returns (text for the model, chunks used as evidence). The evidence
is collected so the critic can check the answer against exactly what was read.
"""

import json

from app.corpus import Chunk, CorpusIndex

GLOSSARY_FILE = "glossary"

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Semantic search over the drug documents. Returns the most relevant passages "
                "with their source file and section. Use `drug` to restrict the search to one drug's files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in natural language."},
                    "drug": {"type": "string", "description": "Optional drug name to filter files, e.g. 'warfarin'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": (
                "Read a full section (e.g. 'contraindications', 'drug interactions', 'dosage', "
                "'warnings', 'mechanism of action') from a drug's documents. Prefer this when the question "
                "asks for a complete list from a known section. If the section is not found, the available "
                "section names are returned so you can retry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {"type": "string", "description": "Drug name, e.g. 'apixaban'."},
                    "section": {"type": "string", "description": "Section name or keyword, e.g. 'contraindications'."},
                },
                "required": ["drug", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_glossary",
            "description": "Look up the definition of a medical or pharmacological term in GLOSSARY.md.",
            "parameters": {
                "type": "object",
                "properties": {"term": {"type": "string", "description": "Term to define, e.g. 'INR' or 'eGFR'."}},
                "required": ["term"],
            },
        },
    },
]


def format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"--- {c.header()}\n{c.text}" for c in chunks) or "No matching passages."


def search_corpus(index: CorpusIndex, query: str, drug: str | None = None) -> tuple[str, list[Chunk]]:
    hits = index.search(query, source_filter=drug)
    if not hits and drug:  # unknown drug name: fall back to an unfiltered search
        hits = index.search(f"{drug} {query}")
    chunks = [chunk for chunk, _ in hits]
    return format_chunks(chunks), chunks


def read_section(index: CorpusIndex, drug: str, section: str) -> tuple[str, list[Chunk]]:
    drug_l, section_l = drug.lower().strip(), section.lower().strip()
    sources = [s for s in index.sources if drug_l in s.lower() and GLOSSARY_FILE not in s.lower()]
    if not sources:
        return f"No document found for '{drug}'. Available files: {', '.join(index.sources)}", []

    chunks = [c for c in index.chunks if c.source in sources and section_l in c.section.lower()]
    if chunks:
        return format_chunks(chunks), chunks
    available = {s: index.sections_of(s) for s in sources}
    return f"Section '{section}' not found. Available sections: {json.dumps(available)}", []


def lookup_glossary(index: CorpusIndex, term: str) -> tuple[str, list[Chunk]]:
    glossary = [c for c in index.chunks if GLOSSARY_FILE in c.source.lower()]
    term_l = term.lower()
    exact = [c for c in glossary if term_l in c.section.lower() or term_l in c.text.lower()]
    if exact:
        # Keep only the lines mentioning the term when the glossary chunk is a long list.
        trimmed = []
        for c in exact[:3]:
            lines = [line for line in c.text.splitlines() if term_l in line.lower()]
            text = "\n".join(lines) if lines and term_l not in c.section.lower() else c.text
            trimmed.append(Chunk(c.source, c.title, c.section, text))
        return format_chunks(trimmed), trimmed
    hits = index.search(term, k=2, source_filter=GLOSSARY_FILE)
    chunks = [c for c, _ in hits]
    return format_chunks(chunks), chunks


def run_tool(index: CorpusIndex, name: str, arguments: dict) -> tuple[str, list[Chunk]]:
    """Dispatch a tool call requested by the model."""
    if name == "search_corpus":
        return search_corpus(index, arguments.get("query", ""), arguments.get("drug"))
    if name == "read_section":
        return read_section(index, arguments.get("drug", ""), arguments.get("section", ""))
    if name == "lookup_glossary":
        return lookup_glossary(index, arguments.get("term", ""))
    return f"Unknown tool: {name}", []

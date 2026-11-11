# Pharma Research Agent

A small agentic research assistant that answers compound questions about a local corpus of
drug labels (DailyMed / MedlinePlus-style markdown files). It plans, retrieves, chooses its own
tools, reviews its draft for grounding, and returns a cited answer with a medical disclaimer.

> Technical exercise only. Not medical advice.

## Quick start

```bash
cp .env.example .env          # fill in your key, base URL and model names
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
```

- UI: http://localhost:8000
- Health: `GET /health` → `200`
- Ask: `POST /ask` with `{"question": "..."}`

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What are the main contraindications for warfarin?"}'
```

Without Docker:

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
python -m app.agent "Compare the contraindications of warfarin and apixaban"   # CLI
pytest -q                                                                     # offline tests
```

## Evaluation

`app/evaluate.py` runs the 4 public evaluation questions and checks that every expected
source file is cited in the answer's `sources` (extra files are allowed).

```bash
python -m app.evaluate                              # agent in-process (needs a key in .env)
python -m app.evaluate --url http://localhost:8000  # against a running API / Docker container
python -m app.evaluate --show-answers --out eval-results.json
```

It prints PASS/FAIL per question with the sources, reviewer verdict and tools used, and exits
with code 1 if any question fails.

## How it works

```
question
   │
   ▼
1. PLAN      LLM splits the question into 1-4 sub-questions (JSON)
   │
   ▼
2. RESEARCH  tool-calling loop (max 6 rounds) — the LLM picks the tools:
   │           • search_corpus(query, drug?)   semantic search, in-memory cosine similarity
   │           • read_section(drug, section)   full section, e.g. all contraindications
   │           • lookup_glossary(term)         definitions from GLOSSARY.md
   │         → draft answer with [file.md] citations
   ▼
3. REVIEW    critic LLM checks every claim against the retrieved evidence:
   │           unsupported claims, clinically ambiguous claims, unanswered parts
   ▼
4. REVISE    only if the critic says "revise": one rewrite using its feedback
   │
   ▼
5. ANSWER    cited answer + sources + disclaimer + plan + tool trace + reviewer notes
```

| File | Role |
|---|---|
| `app/corpus.py` | Loads `corpus/`, splits each file by markdown headings, embeds chunks once, cosine search with numpy |
| `app/tools.py` | The three tools, their JSON schemas, and the dispatcher |
| `app/agent.py` | The pipeline above, as a plain loop over the OpenAI chat API |
| `app/prompts.py` | All prompts in one place |
| `app/llm.py` | OpenAI-compatible client (OpenAI, LiteLLM proxy or Groq) and embedder choice |
| `app/local_embeddings.py` | TF-IDF vectors in numpy, used when no embeddings API is available |
| `app/main.py` | FastAPI: `/health`, `/ask`, serves the UI |
| `app/evaluate.py` | Runs the public evaluation questions and checks the cited sources |
| `static/index.html` | Single-page UI: answer, citations, reviewer badge and notes, plan and tool trace |

## Design choices

- **No agent framework.** The loop is ~40 lines over the OpenAI chat API. Every step is easy to
  read, debug and explain, and the full trace is returned to the client.
- **Section-aware chunking.** Drug labels are organised by sections (Contraindications,
  Drug Interactions…). Chunking by heading keeps each passage self-contained and citable, and
  each chunk is embedded with its file and section name so the vector knows which drug it is about.
- **Two retrieval tools, not one.** Similarity search is good for open questions but can miss
  items of a long list. `read_section` returns a whole section, so "list all contraindications"
  questions get complete answers. The model chooses between them.
- **Drug filter on search.** Multi-drug questions (warfarin + sertraline) are answered by
  searching each drug's files separately, so one drug doesn't crowd out the other.
- **In-memory index.** ~20 files fit in a numpy matrix; no vector database to run.
  The index is built in the background at startup, so `/health` answers immediately.
- **Critic with evidence.** The reviewer sees exactly the passages the agent read, so it can
  catch claims that came from the model's own knowledge.
- **Deterministic safety.** The disclaimer is added by code, and the returned `sources` are
  only files that were both cited and actually read.

## LLM providers

Both providers go through the same OpenAI-compatible SDK (`app/llm.py`), so the agent code is identical.

| | OpenAI / LiteLLM proxy (default) | Groq |
|---|---|---|
| Key | `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for the proxy) | `GROQ_API_KEY` |
| Chat model | `OPENAI_MODEL` (default `gpt-4o-mini`) | `GROQ_MODEL` (default `openai/gpt-oss-120b`) |
| Similarity | API embeddings (`OPENAI_EMBEDDING_MODEL`) | local TF-IDF in numpy (Groq has no embeddings API) |

The provider is `LLM_PROVIDER` if set; otherwise Groq is used only when `GROQ_API_KEY` is set and
`OPENAI_API_KEY` is not. `EMBEDDING_PROVIDER=local` forces the local TF-IDF similarity with either
provider. `GET /health` shows which provider, model and similarity are active.

The local TF-IDF uses words and word pairs, weighted by inverse document frequency and
L2-normalised, so search is still cosine similarity over an in-memory matrix. It works well
here because drug-label questions hinge on exact terms (drug names, section names, "eGFR").

## Configuration

| Variable | Default |
|---|---|
| `CORPUS_DIR` | `corpus/` |
| `MAX_TOOL_ROUNDS` / `TOP_K` / `CHUNK_MAX_CHARS` | `6` / `5` / `1500` |
| `MAX_EVIDENCE_CHARS` | `40000` (evidence sent to the critic; lower it on small rate limits) |

## AI assistance

Built with Claude as a coding assistant. The instructions it followed are in `CLAUDE.md`.

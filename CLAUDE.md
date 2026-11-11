# AI coding configuration

This project was built with Claude as a coding assistant. This file is the instruction
set the assistant followed; it is committed as requested by the assignment.

## Project rules
- Keep the code simple and readable: plain Python, no agent framework, one file per concern
  (`corpus.py`, `tools.py`, `agent.py`, `prompts.py`, `main.py`).
- The agent must answer only from the corpus in `corpus/`. Every fact is cited with its file name.
- The medical disclaimer is added by code (`config.DISCLAIMER`), never left to the model.
- Tool selection is done by the LLM through function calling; never hardcode which tool runs.
- All LLM access goes through `app/llm.py` (OpenAI/LiteLLM or Groq, chosen in `config.py`) and reads `OPENAI_*` or `GROQ_*`
  variables from the environment. Groq has no embeddings API: use the local TF-IDF in `local_embeddings.py`.
- Pin every dependency in `requirements.txt`.
- Tests must run offline (`pytest`): use the fake embedding in `tests/conftest.py` and a scripted fake LLM.

## Workflow
1. Change code.
2. Run `pytest -q`.
3. Check the API manually: `uvicorn app.main:app --reload`, then open http://localhost:8000.

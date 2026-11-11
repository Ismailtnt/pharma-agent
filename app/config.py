"""Settings read from the environment (.env is loaded if present).

Two LLM providers are supported, both through the OpenAI-compatible SDK:
- openai: OpenAI or the LiteLLM proxy (OPENAI_API_KEY, OPENAI_BASE_URL)
- groq:   Groq Cloud (GROQ_API_KEY)
LLM_PROVIDER picks one explicitly; otherwise groq is used only when a Groq key
is set and no OpenAI key is.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_default_provider = "groq" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", _default_provider).lower()

if LLM_PROVIDER == "groq":
    API_KEY = os.getenv("GROQ_API_KEY")
    BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    CHAT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
else:
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")  # None -> api.openai.com
    CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Groq has no embeddings endpoint, so it uses the local TF-IDF similarity by default.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local" if LLM_PROVIDER == "groq" else "openai").lower()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", Path(__file__).resolve().parent.parent / "corpus"))

# Agent limits: keep latency, cost and token usage bounded.
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "6"))
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1500"))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "40000"))
# Conversation context sent to the planner to resolve follow-up questions.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))
HISTORY_ANSWER_CHARS = int(os.getenv("HISTORY_ANSWER_CHARS", "1500"))

DISCLAIMER = (
    "This answer is generated from a limited document set for a technical demonstration. "
    "It is not medical advice. Always consult a qualified healthcare professional."
)

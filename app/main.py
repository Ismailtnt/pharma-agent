"""FastAPI app: GET /health, POST /ask, and the web UI at /."""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import agent, config
from app.corpus import get_index, index_ready

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _warm_up() -> None:
    try:
        index = get_index()
        log.info("Index ready: %d chunks from %d files", len(index.chunks), len(index.sources))
    except Exception:  # first /ask will retry and report the error
        log.exception("Index build failed at startup")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Build the embedding index in the background so the server is up immediately.
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Pharma Research Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_ready": index_ready(),
        "provider": config.LLM_PROVIDER,
        "model": config.CHAT_MODEL,
        "embeddings": config.EMBEDDING_PROVIDER,
    }


@app.post("/ask", response_model=agent.AgentResult)
def ask(request: AskRequest) -> agent.AgentResult:
    # Sync endpoint: FastAPI runs it in a thread pool, so blocking LLM calls are fine.
    try:
        return agent.ask(request.question.strip())
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        log.exception("Agent failed")
        raise HTTPException(status_code=502, detail=f"Agent error: {error}") from error


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

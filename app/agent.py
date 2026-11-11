"""The agent: plan -> research with tools -> draft -> critic review -> (revise) -> cited answer.

Written as a plain loop over the OpenAI chat API rather than a framework, so every
step is visible and returned to the client as a trace.
"""

import json
import re
import time

from pydantic import BaseModel

from app import config, llm, prompts
from app.corpus import Chunk, get_index
from app.tools import TOOL_SPECS, run_tool


class ToolStep(BaseModel):
    tool: str
    arguments: dict
    sources: list[str]


class Turn(BaseModel):
    """One earlier exchange of the conversation, sent back by the client."""
    question: str
    answer: str


class Review(BaseModel):
    verdict: str = "pass"
    unsupported_claims: list[str] = []
    ambiguous_claims: list[str] = []
    missing: list[str] = []
    revised: bool = False


class AgentResult(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources: list[str]
    plan: list[str]
    steps: list[ToolStep]
    review: Review
    disclaimer: str = config.DISCLAIMER
    latency_ms: int = 0


def ask(question: str, history: list[Turn] | None = None) -> AgentResult:
    started = time.perf_counter()
    index = get_index()
    sources = ", ".join(index.sources)

    # Follow-ups ("and for apixaban?") are rewritten as standalone questions by the planner,
    # so the research, review and revision steps never need the conversation itself.
    standalone, plan = make_plan(question, sources, history or [])
    draft, evidence, steps = research(standalone, plan, sources)
    evidence_text = format_evidence(evidence)

    review, feedback = critique(standalone, draft, evidence_text)
    answer = draft
    if review.verdict == "revise":
        answer = revise(standalone, draft, evidence_text, feedback) or draft
        review.revised = True

    return AgentResult(
        question=question,
        standalone_question=standalone,
        answer=answer.strip(),
        sources=cited_sources(answer, evidence),
        plan=plan,
        steps=steps,
        review=review,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


# --- 1. Plan -----------------------------------------------------------------
def make_plan(question: str, sources: str, history: list[Turn]) -> tuple[str, list[str]]:
    """Returns (standalone question, sub-questions)."""
    user = prompts.PLANNER_INPUT.format(history=format_history(history), question=question) if history else question
    data = llm.chat_json(prompts.PLANNER.format(sources=sources), user)
    standalone = str(data.get("standalone_question") or "").strip() or question
    sub_questions = [str(q) for q in data.get("sub_questions", []) if str(q).strip()]
    return standalone, sub_questions or [standalone]


# --- 2. Research (agent-driven tool use) -------------------------------------
def research(question: str, plan: list[str], sources: str) -> tuple[str, list[Chunk], list[ToolStep]]:
    index = get_index()
    plan_text = "\n".join(f"{i}. {q}" for i, q in enumerate(plan, 1))
    messages = [
        {"role": "system", "content": prompts.RESEARCHER.format(sources=sources, plan=plan_text)},
        {"role": "user", "content": question},
    ]
    evidence: list[Chunk] = []
    steps: list[ToolStep] = []

    for _ in range(config.MAX_TOOL_ROUNDS):
        message = llm.chat(messages, tools=TOOL_SPECS)
        if not message.tool_calls:
            return message.content or "", evidence, steps

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ],
        })
        for call in message.tool_calls:
            arguments = llm.parse_json(call.function.arguments or "{}")
            output, chunks = run_tool(index, call.function.name, arguments)
            evidence.extend(chunks)
            steps.append(ToolStep(
                tool=call.function.name,
                arguments=arguments,
                sources=sorted({c.source for c in chunks}),
            ))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    # Tool budget exhausted: force a written answer from what was gathered.
    messages.append({"role": "user", "content": "Stop researching and write the final answer now."})
    message = llm.chat(messages, tools=TOOL_SPECS, tool_choice="none")
    return message.content or "", evidence, steps


# --- 3. Review ---------------------------------------------------------------
def critique(question: str, draft: str, evidence_text: str) -> tuple[Review, str]:
    data = llm.chat_json(
        prompts.CRITIC,
        prompts.CRITIC_INPUT.format(question=question, evidence=evidence_text, draft=draft),
    )
    review = Review(
        verdict="revise" if data.get("verdict") == "revise" else "pass",
        unsupported_claims=[str(x) for x in data.get("unsupported_claims", [])],
        ambiguous_claims=[str(x) for x in data.get("ambiguous_claims", [])],
        missing=[str(x) for x in data.get("missing", [])],
    )
    return review, str(data.get("feedback", ""))


def revise(question: str, draft: str, evidence_text: str, feedback: str) -> str:
    message = llm.chat([
        {"role": "system", "content": prompts.REVISER},
        {"role": "user", "content": prompts.REVISER_INPUT.format(
            question=question, evidence=evidence_text, draft=draft, feedback=feedback)},
    ])
    return message.content or ""


# --- Helpers -----------------------------------------------------------------
def format_history(history: list[Turn]) -> str:
    """Last few turns, answers shortened: enough to resolve follow-ups, cheap in tokens."""
    turns = history[-config.MAX_HISTORY_TURNS:]
    return "\n\n".join(
        f"User: {t.question}\nAssistant: {t.answer[:config.HISTORY_ANSWER_CHARS]}" for t in turns
    )


def format_evidence(chunks: list[Chunk]) -> str:
    """Deduplicated evidence, capped in size, in the same format the model saw."""
    seen, parts, size = set(), [], 0
    for c in chunks:
        key = (c.source, c.section, c.text)
        if key in seen:
            continue
        seen.add(key)
        part = f"--- {c.header()}\n{c.text}"
        if size + len(part) > config.MAX_EVIDENCE_CHARS:
            break
        parts.append(part)
        size += len(part)
    return "\n\n".join(parts) or "(no evidence retrieved)"


def cited_sources(answer: str, evidence: list[Chunk]) -> list[str]:
    """Files cited in the answer that were actually read; falls back to all files read."""
    read = {c.source for c in evidence}
    cited = {name for name in re.findall(r"[\w.\-]+\.(?:md|txt)", answer) if name in read}
    return sorted(cited or read)


if __name__ == "__main__":  # quick CLI: python -m app.agent "your question"
    import sys

    result = ask(" ".join(sys.argv[1:]) or "What are the main contraindications for warfarin?")
    print(result.answer, "\n\nSources:", result.sources, "\nReview:", result.review)
    print("\n" + json.dumps([s.model_dump() for s in result.steps], indent=2))

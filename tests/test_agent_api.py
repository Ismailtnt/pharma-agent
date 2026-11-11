"""End-to-end test of the agent loop and the API with a scripted fake LLM (no network)."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import config, llm
from app.main import app


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def fake_chat(messages, **kwargs):
    if not any(m["role"] == "tool" for m in messages):
        # First round: the model decides to read the contraindications section.
        return SimpleNamespace(content=None, tool_calls=[
            tool_call("1", "read_section", '{"drug": "warfarin", "section": "contraindications"}')])
    return SimpleNamespace(
        content="Warfarin is contraindicated in pregnancy and malignant hypertension [dailymed-warfarin.md].",
        tool_calls=None,
    )


def fake_chat_json(system, user):
    if "plan research" in system:
        return {"sub_questions": ["What are warfarin's contraindications?"], "drugs": ["warfarin"]}
    return {"verdict": "pass", "unsupported_claims": [], "ambiguous_claims": [], "missing": []}


def test_ask_endpoint(index, monkeypatch):
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    response = client.post("/ask", json={"question": "What are the main contraindications for warfarin?"})
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == ["dailymed-warfarin.md"]
    assert data["steps"][0]["tool"] == "read_section"
    assert data["review"]["verdict"] == "pass"
    assert data["disclaimer"] == config.DISCLAIMER


def test_ask_rejects_empty_question(index):
    assert TestClient(app).post("/ask", json={"question": ""}).status_code == 422


def test_follow_up_question_is_resolved_with_history(index, monkeypatch):
    seen = {}

    def planner_with_history(system, user):
        if "plan research" in system:
            seen["planner_input"] = user
            return {"standalone_question": "What are the contraindications of apixaban?",
                    "sub_questions": ["What are apixaban's contraindications?"]}
        return fake_chat_json(system, user)

    def chat(messages, **kwargs):
        seen.setdefault("research_question", messages[1]["content"])
        return fake_chat(messages, **kwargs)

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "chat_json", planner_with_history)

    history = [{"question": "What are the contraindications of warfarin?", "answer": "Pregnancy [dailymed-warfarin.md]."}]
    response = TestClient(app).post("/ask", json={"question": "And for apixaban?", "history": history})

    assert response.status_code == 200
    data = response.json()
    assert "What are the contraindications of warfarin?" in seen["planner_input"]  # history reached the planner
    assert seen["research_question"] == "What are the contraindications of apixaban?"  # research got the rewrite
    assert data["question"] == "And for apixaban?"
    assert data["standalone_question"] == "What are the contraindications of apixaban?"


def test_question_without_history_is_its_own_standalone_question(index, monkeypatch):
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    data = TestClient(app).post("/ask", json={"question": "Warfarin contraindications?"}).json()
    assert data["standalone_question"] == "Warfarin contraindications?"

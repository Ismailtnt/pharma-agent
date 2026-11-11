"""The evaluation checker: expected files must be cited; extra files are allowed."""

from app.evaluate import EVAL_SET, check


def test_eval_set_has_the_four_public_questions():
    assert len(EVAL_SET) == 4
    assert all(case["expected"] for case in EVAL_SET)


def test_check_passes_when_expected_sources_are_cited():
    case = {"question": "q", "expected": ["dailymed-warfarin.md"]}
    result = {"answer": "a", "disclaimer": "d", "sources": ["dailymed-warfarin.md", "medlineplus-warfarin.md"]}
    assert check(case, result)["passed"]


def test_check_fails_and_reports_missing_sources():
    case = {"question": "q", "expected": ["dailymed-warfarin.md", "dailymed-apixaban.md"]}
    row = check(case, {"answer": "a", "disclaimer": "d", "sources": ["dailymed-warfarin.md"]})
    assert not row["passed"]
    assert row["missing"] == ["dailymed-apixaban.md"]


def test_check_handles_a_failed_request():
    assert not check({"question": "q", "expected": ["x.md"]}, {})["passed"]

"""Run the public evaluation questions and check the cited source files.

A question passes when every expected file is in the answer's `sources`
(extra files, e.g. the MedlinePlus monograph of the same drug, are allowed).

    python -m app.evaluate                            # runs the agent in-process
    python -m app.evaluate --url http://localhost:8000   # calls a running API (e.g. the Docker container)
    python -m app.evaluate --out eval-results.json    # also save the full answers
"""

import argparse
import json
import sys
import time
import urllib.request

EVAL_SET = [
    {
        "question": "What are the main contraindications for warfarin?",
        "expected": ["dailymed-warfarin.md"],
    },
    {
        "question": "Compare the contraindications of warfarin and apixaban. "
                    "What conditions are shared and which are unique to each drug?",
        "expected": ["dailymed-warfarin.md", "dailymed-apixaban.md"],
    },
    {
        "question": "A patient currently on warfarin is prescribed sertraline. "
                    "What is the risk of bleeding and what monitoring is recommended?",
        "expected": ["dailymed-warfarin.md", "dailymed-sertraline.md"],
    },
    {
        "question": "A patient with an eGFR of 25 mL/min is being considered for metformin. "
                    "Is metformin contraindicated at this level of renal function, and what is the threshold?",
        "expected": ["dailymed-metformin.md"],
    },
]


def ask_in_process(question: str) -> dict:
    from app import agent  # imported here so --url mode needs no API key or corpus

    return agent.ask(question).model_dump()


def ask_over_http(url: str, question: str) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/ask",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def check(case: dict, result: dict) -> dict:
    """Compare one agent result with the expected source files."""
    sources = result.get("sources", [])
    missing = [f for f in case["expected"] if f not in sources]
    return {
        "question": case["question"],
        "passed": not missing and bool(result.get("answer")) and bool(result.get("disclaimer")),
        "expected": case["expected"],
        "sources": sources,
        "missing": missing,
        "verdict": result.get("review", {}).get("verdict"),
        "revised": result.get("review", {}).get("revised"),
        "tools": [step["tool"] for step in result.get("steps", [])],
        "latency_ms": result.get("latency_ms"),
        "answer": result.get("answer", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="base URL of a running API; omit to run the agent in-process")
    parser.add_argument("--out", help="write the full results (answers included) to this JSON file")
    parser.add_argument("--show-answers", action="store_true", help="print each answer")
    args = parser.parse_args()

    rows = []
    for number, case in enumerate(EVAL_SET, 1):
        print(f"[{number}/{len(EVAL_SET)}] {case['question']}", flush=True)
        started = time.perf_counter()
        try:
            result = ask_over_http(args.url, case["question"]) if args.url else ask_in_process(case["question"])
            row = check(case, result)
        except Exception as error:  # report and keep going with the other questions
            row = {**check(case, {}), "error": str(error)}
        rows.append(row)

        status = "PASS" if row["passed"] else "FAIL"
        seconds = time.perf_counter() - started
        print(f"      {status}  sources={row['sources']}  verdict={row['verdict']}  "
              f"tools={row['tools']}  {seconds:.1f}s")
        if row["missing"]:
            print(f"      missing: {row['missing']}")
        if row.get("error"):
            print(f"      error: {row['error']}")
        if args.show_answers and row["answer"]:
            print("\n" + row["answer"] + "\n")

    passed = sum(row["passed"] for row in rows)
    print(f"\n{passed}/{len(rows)} questions passed")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
        print(f"Results written to {args.out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

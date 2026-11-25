"""Run the evaluation set against analyze_request and report accuracy per field."""

import csv
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from nativ.analysis import AnalysisError, analyze_request
from nativ.models import Request

EVAL_PATH = Path("data/eval_requests.csv")
RESULTS_PATH = Path("data/eval_results.csv")
DELAY_SECONDS = 4  # Keeps us under the Gemini free-tier rate limit.
FIELDS = ("address", "incident_type", "urgency", "missing")


def _normalize_address(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"^\s*(רחוב|רח')\s*", "", value)
    return re.sub(r"[\s,.\-\"']+", " ", value).strip()


def _address_matches(expected: str, actual: str | None) -> bool:
    exp, act = _normalize_address(expected), _normalize_address(actual)
    if not exp or not act:
        return exp == act
    return exp in act or act in exp


def _split_missing(value: str) -> set[str]:
    return {item.strip() for item in value.split(";") if item.strip()}


def _evaluate_row(row: dict[str, str]) -> tuple[dict[str, bool], dict[str, str], str]:
    try:
        analysis = analyze_request(Request(text=row["text"], received_at=datetime.now()))
    except AnalysisError:
        return dict.fromkeys(FIELDS, False), dict.fromkeys(FIELDS, ""), "analysis_error"
    actual = {
        "address": analysis.address or "",
        "incident_type": analysis.incident_type or "",
        "urgency": analysis.urgency or "",
        "missing": ";".join(sorted(analysis.missing_fields)),
    }
    checks = {
        "address": _address_matches(row["expected_address"].strip(), analysis.address),
        "incident_type": row["expected_incident_type"].strip() == actual["incident_type"],
        "urgency": row["expected_urgency"].strip() == actual["urgency"],
        "missing": _split_missing(row["expected_missing"]) == set(analysis.missing_fields),
    }
    return checks, actual, ""


def main() -> None:
    with EVAL_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    correct = dict.fromkeys(FIELDS, 0)
    by_case = defaultdict(lambda: [0, 0])  # case_type -> [fully correct, total]
    results = []

    for i, row in enumerate(rows, start=1):
        checks, actual, error = _evaluate_row(row)
        for field in FIELDS:
            correct[field] += checks[field]
        by_case[row["case_type"]][0] += all(checks.values())
        by_case[row["case_type"]][1] += 1
        marks = " ".join(f"{f}={'OK' if checks[f] else 'X'}" for f in FIELDS)
        print(f"[{i:2}/{len(rows)}] id={row['id']:>3} {marks} {error}")
        results.append({
            "id": row["id"],
            "case_type": row["case_type"],
            "text": row["text"],
            **{f"expected_{f}": row[f"expected_{f}"] for f in FIELDS},
            **{f"actual_{f}": actual[f] for f in FIELDS},
            **{f"ok_{f}": checks[f] for f in FIELDS},
            "error": error,
        })
        if i < len(rows):
            time.sleep(DELAY_SECONDS)

    total = len(rows)
    print("\nAccuracy per field:")
    for field in FIELDS:
        print(f"  {field:<14} {correct[field]:>2}/{total} ({correct[field] / total:.0%})")
    print("\nFully correct requests per case type:")
    for case_type, (ok, n) in by_case.items():
        print(f"  {case_type:<12} {ok}/{n}")

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
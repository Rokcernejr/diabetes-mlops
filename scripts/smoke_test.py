#!/usr/bin/env python3
"""Cross-platform smoke test against a running API instance.

Replaces the old PowerShell scripts so the same checks run on Windows dev
machines, Linux CI, and post-deploy against a port-forwarded cluster:

    python scripts/smoke_test.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

PREDICTION_PAYLOAD = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "time_in_hospital": 7,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 0,
    "number_diagnoses": 9,
    "a1c_result": ">7",
    "max_glu_serum": "None",
    "change": "Ch",
    "diabetesMed": "Yes",
}


def call(base_url: str, path: str, method: str = "GET", body: dict | None = None):
    """Return (status_code, parsed_or_raw_body) for a request."""
    request = urllib.request.Request(base_url + path, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, data=data, timeout=15) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                return response.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except urllib.error.URLError as exc:
        return None, str(exc.reason)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    failures = []

    def check(name: str, ok: bool, detail: str = ""):
        print(
            f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else "")
        )
        if not ok:
            failures.append(name)

    status, body = call(base, "/health")
    check("GET /health returns 200", status == 200, f"status={status}")

    status, body = call(base, "/ready")
    check("GET /ready returns 200 (model loaded)", status == 200, f"status={status}")

    status, body = call(base, "/metrics")
    check("GET /metrics returns 200", status == 200, f"status={status}")

    # A real prediction, not just liveness — this catches train/serve skew
    status, body = call(base, "/predict", method="POST", body=PREDICTION_PAYLOAD)
    prob_ok = (
        status == 200
        and isinstance(body, dict)
        and 0.0 <= body.get("probability", -1) <= 1.0
    )
    check(
        "POST /predict returns a probability", prob_ok, f"status={status} body={body}"
    )

    status, body = call(base, "/model/info")
    check("GET /model/info returns 200", status == 200, f"status={status}")

    if failures:
        print(f"\n{len(failures)} smoke check(s) failed: {', '.join(failures)}")
        return 1
    print("\nAll smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
import time
from pathlib import Path


ERROR_CATEGORIES = (
    "syntax",
    "runtime",
    "timeout",
    "none",
)


def extract_python(text):
    fence = chr(96) * 3
    fenced = re.findall(
        fence + r"python\s*(.*?)" + fence,
        text,
        flags=re.S | re.I,
    )
    if fenced:
        return fenced[-1].strip()

    generic = re.findall(
        fence + r"\s*(.*?)" + fence,
        text,
        flags=re.S,
    )
    return generic[-1].strip() if generic else text.strip()


def compile_ok(code):
    try:
        ast.parse(code)
        compile(code, "<generated>", "exec")
        return True
    except Exception:
        return False


def classify_error(code, result):
    if not compile_ok(code):
        return "syntax"
    if result.get("status") == "timeout":
        return "timeout"
    if result.get("passed"):
        return "none"
    return "runtime"


def run_case(code, harness, timeout_s=5.0):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval.py"
        path.write_text(
            code + "\n\n" + harness,
            encoding="utf-8",
        )
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                ["python", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "status": "timeout",
                "latency_ms": (time.perf_counter() - start) * 1000,
            }

        return {
            "passed": proc.returncode == 0,
            "status": "ok" if proc.returncode == 0 else "runtime_error",
            "latency_ms": (time.perf_counter() - start) * 1000,
            "stderr": proc.stderr,
        }


def evaluate_predictions(predictions, timeout_s=5.0):
    total = len(predictions)
    if not total:
        return {
            "status": "not_yet_evaluated",
            "eval_examples": 0,
            "compile_rate": None,
            "pass_rate": None,
            "execution_success_rate": None,
            "inference_latency_ms": None,
            "error_distribution": {},
        }

    compiled = 0
    executed = 0
    tested = 0
    passed = 0
    latencies = []
    errors = {key: 0 for key in ERROR_CATEGORIES}

    for item in predictions:
        code = extract_python(
            str(item.get("prediction", ""))
        )

        if not compile_ok(code):
            errors["syntax"] += 1
            continue

        compiled += 1
        harness = str(item.get("harness", ""))

        if not harness:
            errors["none"] += 1
            continue

        tested += 1
        result = run_case(
            code,
            harness,
            timeout_s=timeout_s,
        )
        latencies.append(
            result["latency_ms"]
        )

        if result["passed"]:
            passed += 1
            errors["none"] += 1
            executed += 1
        elif result["status"] == "timeout":
            errors["timeout"] += 1
        else:
            errors["runtime"] += 1

    return {
        "status": "measured",
        "eval_examples": total,
        "compile_rate": compiled / total,
        "execution_success_rate": executed / tested if tested else None,
        "pass_rate": passed / tested if tested else None,
        "tested_examples": tested,
        "inference_latency_ms": (
            sum(latencies) / len(latencies)
            if latencies
            else None
        ),
        "error_distribution": {
            key: value
            for key, value in errors.items()
            if value
        },
    }

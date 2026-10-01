from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def audit(train_rows, eval_rows):
    train_ids = {str(row.get("problem_id", "")) for row in train_rows if row.get("problem_id")}
    eval_ids = {str(row.get("problem_id", "")) for row in eval_rows if row.get("problem_id")}
    train_text = {normalize(row.get("prompt", row.get("text", ""))) for row in train_rows}
    eval_text = {normalize(row.get("prompt", row.get("text", ""))) for row in eval_rows}
    id_overlap = sorted(train_ids & eval_ids)
    text_overlap = sorted(train_text & eval_text)
    return {
        "status": "overlap_detected" if id_overlap or text_overlap else "no_exact_overlap_detected",
        "certification": "not_certified",
        "id_overlap_count": len(id_overlap),
        "text_overlap_count": len(text_overlap),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", default="results/leakage_audit.json")
    args = parser.parse_args()

    report = audit(
        read_jsonl(args.train),
        read_jsonl(args.eval),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

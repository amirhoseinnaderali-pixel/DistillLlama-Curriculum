import argparse
import json
from pathlib import Path

from cgkd_cr.evaluation import evaluate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", default=None)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    metrics = evaluate_predictions(
        rows,
        timeout_s=args.timeout_s,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    if args.result:
        result_path = Path(args.result)
        result = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        result.update(
            {
                "eval_examples": metrics.get("eval_examples"),
                "compile_rate": metrics.get("compile_rate"),
                "solve_rate": metrics.get("pass_rate"),
                "pass_rate": metrics.get("pass_rate"),
                "inference_latency_ms": metrics.get(
                    "inference_latency_ms"
                ),
                "status": metrics.get(
                    "status",
                    "not_yet_evaluated",
                ),
                "metadata": {
                    **result.get("metadata", {}),
                    "evaluation": metrics,
                },
            }
        )
        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

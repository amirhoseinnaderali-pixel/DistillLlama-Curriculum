import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    stages = manifest.get("stage_records", [])
    labels = [x.get("stage", "") for x in stages]
    losses = []
    for x in stages:
        metrics = x.get("metrics", {})
        losses.append(metrics.get("train_loss"))
    valid = [(label, loss) for label, loss in zip(labels, losses) if loss is not None]

    if not valid:
        raise SystemExit("No stage-level train_loss values were recorded.")

    x, y = zip(*valid)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(y) + 1), y, marker="o")
    plt.xticks(range(1, len(y) + 1), x)
    plt.xlabel("Curriculum stage")
    plt.ylabel("Training loss")
    plt.title("CGKD-CR Stage Training Loss")
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()

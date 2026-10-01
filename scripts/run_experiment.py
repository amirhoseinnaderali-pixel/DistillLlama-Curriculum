import argparse
import json
import subprocess
from pathlib import Path

from cgkd_cr.config import load_config, resolve_stage_order, save_json, stable_hash
from cgkd_cr.results import ExperimentResult
from cgkd_cr.training import train_experiment


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["config_path"] = args.config

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_id": cfg["experiment_id"],
                    "method": cfg["method"],
                    "stage_order": resolve_stage_order(cfg),
                    "teacher": cfg.get("distillation", {}).get("teacher_model"),
                    "training": cfg["training"],
                },
                indent=2,
            )
        )
        return

    manifest = train_experiment(cfg)
    stages = manifest.get("stage_records", [])

    last = stages[-1] if stages else {}
    gpu_values = [
        x.get("peak_gpu_memory_gb")
        for x in stages
        if x.get("peak_gpu_memory_gb") is not None
    ]

    train_tokens = manifest.get("train_tokens")
    seconds = float(manifest.get("training_time_hours", 0.0)) * 3600
    tokens_per_second = (
        float(train_tokens) / seconds
        if train_tokens and seconds > 0
        else None
    )

    result = ExperimentResult(
        experiment_id=cfg["experiment_id"],
        model=cfg["model"]["base_model"],
        method=cfg["method"],
        stage_order=manifest["stage_order"],
        teacher_models=manifest.get("teacher_models", []),
        train_examples=int(manifest.get("train_examples", 0)),
        train_tokens=train_tokens,
        eval_examples=None,
        teacher_tokens=manifest.get("teacher_tokens"),
        optimizer_steps=manifest.get("optimizer_steps"),
        training_time_hours=manifest.get("training_time_hours"),
        tokens_per_second=tokens_per_second,
        gpu_memory_gb=max(gpu_values) if gpu_values else None,
        parameter_count=last.get("parameter_count"),
        trainable_parameter_count=last.get("trainable_parameter_count"),
        seed=int(cfg.get("seed", 0)),
        config_hash=stable_hash(cfg),
        git_commit=current_git_commit(),
        status="pending_experiment",
        metadata={
            "budget": cfg.get("budget", {}),
            "training_metrics": stages,
        },
    )

    save_json(
        Path(cfg["runtime"]["output_root"]) / "result.json",
        result.to_dict(),
    )


if __name__ == "__main__":
    main()

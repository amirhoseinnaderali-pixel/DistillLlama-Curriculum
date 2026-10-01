from __future__ import annotations

import json
import time
from pathlib import Path

from .checkpoints import write_lineage
from .config import resolve_stage_order, save_json, set_seed
from .data import STAGES, format_supervised_example, load_stage_dataset
from .distillation import generate_records


def load_model(model_name, max_seq_length):
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Unsloth is required for training.") from exc

    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )
    return FastLanguageModel, model, tokenizer


def attach_lora(FastLanguageModel, model, cfg):
    return FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg.get("lora_alpha", cfg["lora_r"] * 2),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        use_gradient_checkpointing="unsloth",
    )


def attach_previous_adapter(model, previous_checkpoint):
    from peft import PeftModel
    return PeftModel.from_pretrained(
        model,
        previous_checkpoint,
        is_trainable=True,
    )


def make_trainer(model, tokenizer, dataset, cfg):
    from transformers import TrainingArguments
    from trl import SFTTrainer

    try:
        from trl import SFTConfig

        args = SFTConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg.get("warmup_steps", 0),
            fp16=cfg.get("fp16", True),
            logging_steps=cfg.get("logging_steps", 10),
            optim=cfg.get("optim", "adamw_8bit"),
            weight_decay=cfg.get("weight_decay", 0.01),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            save_strategy=cfg.get("save_strategy", "epoch"),
            save_total_limit=cfg.get("save_total_limit", 2),
            report_to="none",
            dataset_text_field="text",
            max_length=cfg["max_seq_length"],
        )
        return SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=args,
            processing_class=tokenizer,
        )
    except (ImportError, TypeError):
        args = TrainingArguments(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg.get("warmup_steps", 0),
            fp16=cfg.get("fp16", True),
            logging_steps=cfg.get("logging_steps", 10),
            optim=cfg.get("optim", "adamw_8bit"),
            weight_decay=cfg.get("weight_decay", 0.01),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            save_strategy=cfg.get("save_strategy", "epoch"),
            save_total_limit=cfg.get("save_total_limit", 2),
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to="none",
        )
        return SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=args,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
        )


def token_count(dataset, tokenizer, batch_size=32):
    total = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]["text"]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
        )
        total += sum(len(ids) for ids in encoded["input_ids"])
    return int(total)


def cap_dataset_by_tokens(dataset, tokenizer, max_tokens):
    if max_tokens is None:
        return dataset

    kept=[]
    total=0
    for index, text in enumerate(dataset["text"]):
        length=len(tokenizer(str(text), add_special_tokens=True)["input_ids"])
        if kept and total + length > int(max_tokens):
            break
        kept.append(index)
        total += length

    if not kept:
        raise ValueError("max_train_tokens is smaller than the first example.")

    return dataset.select(kept)


def teacher_token_count(records, tokenizer):
    total = 0
    for row in records:
        response = str(row.get("teacher_response", ""))
        if response:
            total += len(
                tokenizer(
                    response,
                    add_special_tokens=True,
                )["input_ids"]
            )
    return int(total)


def write_records_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_stage_dataset(stage, config, teacher_enabled, provenance_root):
    ds = load_stage_dataset(stage, config)

    if not teacher_enabled:
        return ds.map(lambda x: format_supervised_example(stage, x)), []

    override = config.get("stage_overrides", {}).get(stage, {})
    records = generate_records(
        ds,
        stage=stage,
        teacher_model=config["distillation"]["teacher_model"],
        source_dataset=override.get(
            "dataset",
            STAGES[stage]["dataset"],
        ),
        max_samples=config["distillation"].get("max_samples"),
        max_tokens=int(
            config["distillation"].get("max_tokens", 1024)
        ),
        temperature=float(
            config["distillation"].get("temperature", 0.7)
        ),
    )

    if not records:
        raise RuntimeError(
            f"Teacher generation produced no records for stage {stage}."
        )

    write_records_jsonl(
        records,
        Path(provenance_root) / f"{stage}.jsonl",
    )

    from datasets import Dataset

    return Dataset.from_list(records), records


def stage_training_config(config, stage, output_dir):
    cfg = dict(config["training"]["stage_defaults"])
    cfg.update(config.get("stage_training", {}).get(stage, {}))
    cfg["output_dir"] = str(output_dir)
    return cfg


def train_one(
    *,
    base_model,
    previous_checkpoint,
    stage,
    dataset,
    teacher_records,
    config,
    cfg,
    teacher_enabled,
):
    import torch

    FastLanguageModel, model, tokenizer = load_model(
        base_model,
        cfg["max_seq_length"],
    )

    if previous_checkpoint:
        model = attach_previous_adapter(
            model,
            previous_checkpoint,
        )
    else:
        model = attach_lora(
            FastLanguageModel,
            model,
            cfg,
        )

    max_train_tokens=config.get("budget",{}).get("max_train_tokens")
    dataset=cap_dataset_by_tokens(dataset, tokenizer, max_train_tokens)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_tokens = token_count(
        dataset,
        tokenizer,
    )
    teacher_tokens = (
        teacher_token_count(
            teacher_records,
            tokenizer,
        )
        if teacher_enabled
        else None
    )

    trainer = make_trainer(
        model,
        tokenizer,
        dataset,
        cfg,
    )
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    peak_gpu_gb = None
    if torch.cuda.is_available():
        peak_gpu_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )
    trainable_parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    train_loss = result.metrics.get("train_loss")
    optimizer_steps = int(
        getattr(
            trainer.state,
            "global_step",
            0,
        )
    )
    runtime_seconds = float(
        result.metrics.get(
            "train_runtime",
            elapsed,
        )
    )

    stage_record = {
        "stage": stage,
        "budget_mode": config.get("budget",{}).get("mode","report_only"),
        "max_train_tokens": config.get("budget",{}).get("max_train_tokens"),
        "examples": len(dataset),
        "train_tokens": train_tokens,
        "teacher_tokens": teacher_tokens,
        "optimizer_steps": optimizer_steps,
        "train_loss": train_loss,
        "training_time_hours": runtime_seconds / 3600,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "metrics": result.metrics,
    }

    save_json(
        output_dir / "training_metrics.json",
        stage_record,
    )
    save_json(
        output_dir / "training_log.json",
        {"log_history": trainer.state.log_history},
    )

    write_lineage(
        output_dir / "checkpoint_lineage.json",
        base_model=base_model,
        previous_checkpoint=previous_checkpoint,
        stage=stage,
        dataset=config.get(
            "stage_overrides",
            {},
        ).get(stage, {}).get(
            "dataset",
            STAGES.get(stage, {}).get(
                "dataset",
                "mixed",
            ),
        ),
        teacher=(
            config["distillation"]["teacher_model"]
            if teacher_enabled
            else None
        ),
        training_config=config.get(
            "config_path",
            "unknown",
        ),
        seed=int(config.get("seed", 0)),
    )

    return stage_record


def train_experiment(config):
    set_seed(
        int(
            config.get(
                "seed",
                0,
            )
        )
    )

    base_model = config["model"]["base_model"]
    order = resolve_stage_order(config)
    teacher_enabled = bool(
        config.get(
            "distillation",
            {},
        ).get(
            "enabled"
        )
    )

    root = Path(
        config["runtime"]["output_root"]
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance_root = root / "distillation"
    previous = None
    records = []
    started = time.perf_counter()

    if order == ["mixed"]:
        from datasets import concatenate_datasets

        parts = []
        all_teacher_records = []

        for stage in config["curriculum"]["stages"]:
            dataset, teacher_records = prepare_stage_dataset(
                stage,
                config,
                teacher_enabled,
                provenance_root,
            )
            parts.append(dataset)
            all_teacher_records.extend(
                teacher_records
            )

        train_dataset = concatenate_datasets(parts)
        cfg = stage_training_config(
            config,
            "mixed",
            root / "mixed",
        )

        record = train_one(
            base_model=base_model,
            previous_checkpoint=None,
            stage="mixed",
            dataset=train_dataset,
            teacher_records=all_teacher_records,
            config=config,
            cfg=cfg,
            teacher_enabled=teacher_enabled,
        )
        records.append(record)
        previous = cfg["output_dir"]

    else:
        for index, stage in enumerate(order):
            dataset, teacher_records = prepare_stage_dataset(
                stage,
                config,
                teacher_enabled,
                provenance_root,
            )
            cfg = stage_training_config(
                config,
                stage,
                root / f"stage{index + 1}-{stage}",
            )

            record = train_one(
                base_model=base_model,
                previous_checkpoint=previous,
                stage=stage,
                dataset=dataset,
                teacher_records=teacher_records,
                config=config,
                cfg=cfg,
                teacher_enabled=teacher_enabled,
            )
            records.append(record)
            previous = cfg["output_dir"]

    total_seconds = time.perf_counter() - started

    manifest = {
        "experiment_id": config["experiment_id"],
        "method": config["method"],
        "seed": int(config.get("seed", 0)),
        "stage_order": order,
        "final_model": previous,
        "stage_records": records,
        "train_examples": sum(
            record["examples"]
            for record in records
        ),
        "train_tokens": sum(
            int(
                record.get(
                    "train_tokens",
                    0,
                )
                or 0
            )
            for record in records
        ),
        "teacher_tokens": (
            sum(
                int(
                    record.get(
                        "teacher_tokens",
                        0,
                    )
                    or 0
                )
                for record in records
            )
            if teacher_enabled
            else None
        ),
        "optimizer_steps": sum(
            int(
                record.get(
                    "optimizer_steps",
                    0,
                )
                or 0
            )
            for record in records
        ),
        "training_time_hours": total_seconds / 3600,
        "teacher_models": (
            [config["distillation"]["teacher_model"]]
            if teacher_enabled
            else []
        ),
    }

    save_json(
        root / "run_manifest.json",
        manifest,
    )
    return manifest

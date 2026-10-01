import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM


def extract_code(text):
    text = (text or "").strip()
    fence = chr(96) * 3
    pf = fence + "python"
    if pf in text:
        return text.split(pf, 1)[1].split(fence, 1)[0].strip()
    if fence in text:
        parts = text.split(fence)
        if len(parts) >= 3:
            return parts[1].strip()
    return text


def execute(code, input_data, timeout=8):
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
            "seconds": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nExecution timeout",
            "timeout": True,
            "seconds": time.perf_counter() - started,
        }


def evaluate_code(code, tests):
    rows = []
    for test in tests:
        result = execute(code, test["input"])
        actual = result["stdout"].strip()
        expected = str(test["expected_output"]).strip()
        rows.append(
            {
                "passed": result["ok"] and not result["timeout"] and actual == expected,
                "expected": expected,
                "actual": actual,
                "stderr": result["stderr"],
                "timeout": result["timeout"],
            }
        )
    return {
        "passed_count": sum(row["passed"] for row in rows),
        "total": len(rows),
        "all_passed": bool(rows) and all(row["passed"] for row in rows),
        "rows": rows,
    }


def load_checkpoint(base_model, checkpoint):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    if checkpoint == "base":
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16,
            is_trainable=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model.eval()
    return model, tokenizer


def make_prompt(tokenizer, problem):
    system = (
        "You are an expert Python programmer. "
        "Return only a complete Python 3 program that solves the task. "
        "The program must read stdin and print the required output."
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": problem},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + system
        + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
        + problem
        + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


def evaluate(args):
    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]

    model, tokenizer = load_checkpoint(args.base_model, args.checkpoint)
    device = next(model.parameters()).device

    task_rows = []
    total_seconds = 0.0
    passed_tasks = 0
    passed_tests = 0
    total_tests = 0

    for task in tqdm(tasks, desc=args.checkpoint):
        inputs = make_prompt(tokenizer, task["problem"]).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_seconds = time.perf_counter() - started
        total_seconds += generation_seconds

        text = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        )
        code = extract_code(text)
        evaluation = evaluate_code(code, task["tests"])
        passed_tasks += int(evaluation["all_passed"])
        passed_tests += evaluation["passed_count"]
        total_tests += evaluation["total"]

        task_rows.append(
            {
                "task_id": task["id"],
                "generated_code": code,
                "generation_seconds": generation_seconds,
                "evaluation": evaluation,
            }
        )

    n = len(tasks)
    result = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "tasks": n,
        "task_pass_rate": passed_tasks / n if n else 0.0,
        "test_pass_rate": passed_tests / total_tests if total_tests else 0.0,
        "mean_generation_seconds": total_seconds / n if n else 0.0,
        "rows": task_rows,
    }

    out = Path(args.output or ("results/" + Path(args.checkpoint).name + ".json"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit")
    parser.add_argument("--tasks", default="benchmarks/coding_tasks.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate(args)

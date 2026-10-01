# DistillLlama Curriculum Fine-tuning

This project studies a four-stage curriculum for DeepSeek-R1-Distill-Llama-8B using LoRA, with an optional knowledge-distillation path.

## Research question

> Does progressive curriculum fine-tuning improve executable coding performance, and which stages contribute measurable gains?

## Original curriculum

`Foundation → Algorithms → Debugging → Advanced reasoning`

The notebook changes several factors between stages, including datasets, LoRA rank, learning rate, epochs and sequence length.

## Evidence status

The README historically listed indicative metrics such as ~38%, ~57% and ~66%. The checked-in notebook does not contain matching executed benchmark outputs or a reproducible calculation for those values.

Therefore those figures are not treated as measured results.

## Controlled evaluation added in this branch

`benchmarks/coding_tasks.json` contains eight executable Python tasks with multiple tests each.

The evaluator compares:
- frozen base model
- Stage 1 adapter
- Stage 2 adapter
- Stage 3 adapter
- Stage 4 adapter

Primary metrics:
- task pass rate
- test-case pass rate

Secondary metric:
- generation latency

## Run

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_checkpoints.py --checkpoint base --limit 8
python scripts/evaluate_checkpoints.py --checkpoint ./stage1-foundation --limit 8
python scripts/evaluate_checkpoints.py --checkpoint ./stage2-algorithms --limit 8
python scripts/evaluate_checkpoints.py --checkpoint ./stage3-debugging --limit 8
python scripts/evaluate_checkpoints.py --checkpoint ./stage4-advanced --limit 8
python scripts/analyze_checkpoints.py results/base.json results/stage1-foundation.json results/stage2-algorithms.json results/stage3-debugging.json results/stage4-advanced.json
```

`AutoPeftModelForCausalLM` is used for adapter checkpoints so the adapter metadata can determine its base configuration.

## Stronger causal experiment

Stage-wise gains do not by themselves establish that curriculum ordering is the cause. The next experiment should compare this curriculum against pooled-data SFT with matched:
- training tokens
- optimizer
- learning-rate schedule
- LoRA target modules
- total update steps

## Status

Implemented:
- curriculum training prototype
- stage-wise executable benchmark
- checkpoint evaluation
- result analysis
- methodological audit

Not yet demonstrated:
- a measured stage-by-stage improvement
- a matched-budget curriculum-vs-pooled baseline

## Research trajectory

`PEFT → curriculum adaptation → knowledge distillation → efficient reasoning under compute constraints`
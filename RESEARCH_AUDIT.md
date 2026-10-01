# Research Audit — DistillLlama Curriculum

## Research question

> Does a multi-stage curriculum improve executable code-generation performance over a frozen student baseline, and which curriculum stage contributes measurable gains?

## Original design

The notebook chains four LoRA stages on DeepSeek-R1-Distill-Llama-8B:
1. Foundation — Python instruction data
2. Algorithms — APPS / algorithmic data
3. Debugging — Code-Feedback
4. Advanced — Magicoder-Evol-Instruct

Stage configurations change multiple factors simultaneously: dataset, LoRA rank, learning rate, epochs, sequence length and batch size.

## Evidence audit

The README contains indicative metrics such as ~38%, ~57% and ~66%, but the checked-in notebook does not contain corresponding executed evaluation outputs or a reproducible metric computation for those numbers.

Therefore those values are **not treated as measured results**.

The notebook is better described as an implemented training prototype.

## Main methodological gaps

1. No fixed held-out benchmark is evaluated before and after each stage.
2. No base-model control is reported on the same executable tasks.
3. Curriculum order is confounded with changing hyperparameters and datasets.
4. There is no matched-budget non-curriculum baseline.
5. The optional teacher distillation path is a separate factor and should not be mixed into the curriculum claim.
6. The final sample-generation cell is qualitative rather than an objective task evaluator.

## Controlled redesign

First establish a stage-wise intervention study:
- Base student
- Stage 1 adapter
- Stage 2 adapter
- Stage 3 adapter
- Stage 4 adapter

Use a fixed executable coding benchmark with hidden tests. Report:
- task pass rate
- test-case pass rate
- parse/compile success
- generation latency
- model checkpoint

Then run the stronger causal comparison:
- curriculum sequence
- pooled-data SFT under the same token/step budget
- matched LoRA configuration

## Hypothesis

> Curriculum stages that progressively move from general coding to algorithms and debugging will improve executable coding performance, with later stages producing incremental but potentially diminishing gains.

This is falsifiable.
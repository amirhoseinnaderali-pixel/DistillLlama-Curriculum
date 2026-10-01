# CGKD-CR

## Curriculum-Guided Knowledge Distillation for Efficient Code Reasoning in Small Language Models

CGKD-CR is a research-grade experimental framework for testing whether curriculum-guided training and response-level teacher supervision improve code reasoning in a small language model.

Research status: the repository currently defines the protocol and implementation. No new benchmark result from this refactor is presented as validated evidence.

## Abstract

The project converts a four-stage coding fine-tuning notebook into a controlled experimental framework. The central design separates curriculum and teacher supervision so that the effects of supervised fine-tuning, response-level distillation, curriculum ordering, and their combination can be measured under explicitly reported training budgets.

## Research Question

Can curriculum-guided fine-tuning combined with response-level teacher supervision improve code reasoning in a small language model compared with matched single-stage SFT and with each component evaluated separately?

## Hypotheses

- H1 — Curriculum: staged training improves code-reasoning performance over single-stage mixed-data SFT under a matched budget.
- H2 — Distillation: teacher-generated target responses improve reasoning quality relative to ordinary SFT when student data and budget are controlled.
- H3 — Interaction: curriculum and teacher supervision provide complementary effects.
- H4 — Stage ordering: changing the order of capability-specific stages changes final performance.
- H5 — Efficiency: the student provides an empirically useful quality/inference-cost trade-off relative to stronger reference models.

All hypotheses remain hypotheses until measured.

## Motivation

The legacy repository already combined LoRA fine-tuning, four capability stages, code datasets, and teacher-generated supervision. The missing research layer was attribution: without explicit controls, a final score cannot tell whether improvement came from data volume, data order, teacher supervision, or their interaction.

## Method

The student is unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit with LoRA/PEFT. The teacher pathway is response-level supervision: a teacher generates target text and the student is trained on that target. It is not logit-level knowledge distillation.

The legacy curriculum is:

Foundation → Algorithms → Debugging → Advanced Reasoning

## Curriculum Design

| Stage | Dataset | Intended capability |
|---|---|---|
| Foundation | iamtarun/python_code_instructions_18k_alpaca | instruction following and basic code generation |
| Algorithms | codeparrot/apps | algorithmic problem solving |
| Debugging | m-a-p/Code-Feedback | bug diagnosis and repair |
| Advanced | ise-uiuc/Magicoder-Evol-Instruct-110K | broader coding and reasoning |

Research variants include ordered, random, reverse, and partial/leave-one-stage-out curricula.

## Knowledge Distillation

Teacher metadata is recorded with each generated sample: source dataset, stable problem/example ID, stage, teacher model, generation parameters, teacher response, source fields, and quality/status metadata.

## Experimental Setup

### Central 2 × 2 matrix

| Method | Curriculum | Teacher supervision | LoRA |
|---|---:|---:|---:|
| SFT baseline | No | No | Yes |
| Distillation baseline | No | Yes | Yes |
| Curriculum | Yes | No | Yes |
| CGKD-CR | Yes | Yes | Yes |

## Baselines

- configs/sft_baseline.yaml
- configs/distillation.yaml
- configs/curriculum.yaml
- configs/cgkd_cr.yaml

## Ablation Studies

Current configs also provide random and reverse curriculum. The next documented ablations are stage removal, stage-order sensitivity, one versus multiple teachers, teacher prompt changes, teacher strength, LoRA rank, learning rate, epochs, and data-volume controls.

See docs/ablation_plan.md.

## Evaluation

Training metrics: training loss, optimizer/global steps, training time, training token count, peak GPU memory.

Code reasoning metrics: syntax/compile success, execution/test pass, solve/pass rate when a harness exists, first-pass correctness, pass@k where multiple generations are available, and capability-specific scores.

Efficiency metrics: parameter count, trainable parameter count, training time, inference latency, tokens/sec when measured, and GPU memory.

Evaluation inputs are expected to be unseen. The local ioi_multi_view.json is treated as training-source material unless an explicit contamination audit establishes otherwise.

## Results

No new scientific result is claimed by this refactor. Result records use results/schema.json. Unmeasured fields remain null, and planned runs remain Pending evaluation.

## Capability Transfer

The protocol evaluates capability slices before and after curriculum stages to identify transfer and forgetting. See docs/capability_transfer.md.

## Error Analysis

The taxonomy includes algorithmic misunderstanding, reasoning failure, hallucinated API/code, syntax, compilation, runtime, wrong algorithm, off-by-one, incomplete solution, timeout, memory failure, and training-style overfitting. See docs/error_analysis.md.

## Compute Efficiency

Budget accounting reports training examples, training tokens, teacher target tokens when consistently measurable, optimizer steps, wall-clock time, GPU memory, and trainable parameters. See docs/compute_budget.md.

## Reproducibility

Install: pip install -r requirements.txt

Tests: PYTHONPATH=src pytest -q

Dry run: PYTHONPATH=src python scripts/run_experiment.py --config configs/cgkd_cr.yaml --dry-run

Train: PYTHONPATH=src python scripts/train.py --config configs/cgkd_cr.yaml

Generate teacher data: PYTHONPATH=src python scripts/generate_distillation_data.py --config configs/distillation.yaml --stage algorithms --teacher gemini-2.5-pro --max-samples 200 --output runs/distillation/algorithms.jsonl

Evaluate: PYTHONPATH=src python scripts/evaluate.py --predictions runs/predictions.jsonl --output runs/evaluation.json

Plot stage losses: python scripts/plot_learning_curves.py --manifest runs/cgkd_cr_v1/run_manifest.json --output results/cgkd_cr_loss.png

Run the leakage and credential audits before publishing results.

## Limitations

- response-level rather than logit-level KD
- evaluation quality depends on the supplied benchmark/test runner
- contamination must be audited against the exact train/eval artifacts
- external teacher/API versions may drift
- repeated seeds are required for statistical claims
- arbitrary generated-code evaluation should run in an isolated environment

## Scientific Integrity

Never fabricate benchmark scores, improvements, significance tests, teacher/student comparisons, efficiency gains, or novelty claims.

Use statuses: measured, pending_experiment, not_yet_evaluated, failed.

## Citation

Add a formal citation only after project metadata and experimental results are finalized.

## Repository Rename

Target repository name: curriculum-guided-knowledge-distillation-code-reasoning

The available GitHub write interface can modify branches and repository contents but does not expose the repository-rename endpoint. The actual GitHub rename still needs to be performed from repository Settings by the owner.

## Repository Layout

- src/cgkd_cr/ — core framework
- scripts/ — reproducible entry points
- configs/ — experiment conditions
- docs/ — research protocol and audits
- results/ — generated result artifacts
- tests/ — unit tests
- Colab_Curriculum_Finetune.ipynb — optional notebook interface

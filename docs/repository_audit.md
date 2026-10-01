# Repository Audit

## Scope

The default branch was inspected before the research refactor:

- README.md
- Colab_Curriculum_Finetune.ipynb
- ioi_multi_view.json
- notebook training logic
- curriculum stage definitions
- LoRA configuration
- teacher integration
- distillation flow
- checkpoint behavior
- evaluation surface

The original repository was primarily a single Colab notebook plus the local IOI multi-view
JSON dataset.

## Base model

The original notebook loaded:

unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit

with 4-bit loading and LoRA targets covering attention and MLP projection modules.

## Curriculum

The implemented four-stage sequence was:

1. Foundation
2. Algorithms
3. Debugging
4. Advanced reasoning

The stage data sources were Python instruction data, APPS, Code-Feedback, and Magicoder
Evol-Instruct, with optional local IOI multi-view data in the algorithms path.

## Distillation

The original notebook supported several teacher providers/models and generated teacher
responses that were inserted into student training text.

The legacy implementation did not expose teacher supervision as an orthogonal experimental
factor. It also lacked persistent per-sample provenance for all generated records.

## Evaluation

The original repository did not provide a standardized held-out benchmark runner with
machine-readable result records, capability slices, or a formal contamination audit.

## Reproducibility

The notebook did not provide a research-run manifest covering every seed, stage transition,
teacher configuration, and training-budget quantity.

## Checkpoint behavior

The original training logic used resume-from-checkpoint behavior as part of the notebook
pipeline. That was not a clean scientific checkpoint-lineage protocol for comparing
independent baselines.

## Research interpretation

The original implementation combined several established techniques. The repository was
therefore not treated as evidence of novelty merely because curriculum, LoRA, and teacher
supervision appeared together.

## Existing result claims

The legacy README contained indicative/sample metrics. They are not carried forward as
validated results because the original repository did not contain enough run metadata to
reproduce them.

## Current refactor status

The research branch now adds configuration-driven experiments, an explicit 2 x 2 matrix,
stage-order variants, response-level teacher provenance, checkpoint lineage, result
serialization, evaluation utilities, tests, CI, and research documentation.

The branch still requires real Tier-1 runs and a certified held-out benchmark before any
performance conclusion is written.

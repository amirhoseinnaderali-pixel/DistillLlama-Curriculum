# Prioritized Experimental Plan

## Tier 1 — Core causal matrix

Run with the same student, evaluation split, prompt format, seed policy, and explicit
training/token budgets:

1. SFT baseline
2. Distillation baseline
3. Ordered curriculum
4. CGKD-CR

These four cells answer the primary component and interaction questions.

## Tier 2 — Curriculum causality

5. Random curriculum
6. Reverse curriculum
7. Remove one stage at a time

Current leave-one-stage-out examples are:
- configs/curriculum_no_debugging.yaml
- configs/curriculum_no_advanced.yaml

Add the remaining stage-removal conditions using the same protocol.

## Tier 3 — Teacher effects

8. One teacher vs multiple teachers
9. Weaker vs stronger teacher
10. Teacher prompt/temperature sensitivity

Teacher generation settings and per-sample provenance must remain fixed except for the
factor being studied.

## Tier 4 — Training sensitivity

11. LoRA rank
12. Learning rate
13. Data volume
14. Epochs

Only vary one factor at a time after the Tier-1 matrix is established.

## Budget policy

The configs declare budget.mode and budget.max_train_tokens. Setting a common
max_train_tokens enables an explicit token-budget control; leaving it null records the
observed budget without enforcing a cap.

Report examples, tokens, teacher target tokens, optimizer steps, wall-clock time, and GPU
memory for every run.

## Repetition policy

Repeat Tier-1 conditions with additional seeds when compute allows. Report mean and
dispersion rather than a single-run number when repeated runs exist.

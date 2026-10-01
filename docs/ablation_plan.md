# Prioritized Experimental Plan

## Tier 1 — Core causal matrix

Run with the same student, evaluation split, prompt format, and matched training/token budgets:

1. SFT baseline
2. Distillation baseline
3. Ordered curriculum
4. CGKD-CR

These four cells answer the primary component and interaction questions.

## Tier 2 — Curriculum causality

5. Random curriculum
6. Reverse curriculum
7. Remove one stage at a time

These experiments test whether the ordered progression matters and which stage contributes.

## Tier 3 — Teacher effects

8. One teacher vs multiple teachers
9. Weaker vs stronger teacher
10. Teacher prompt/temperature sensitivity

These experiments test whether the source of teacher supervision changes outcomes.

## Tier 4 — Training sensitivity

11. LoRA rank
12. Learning rate
13. Data volume
14. Epochs

Only vary one factor at a time after the core matrix is established.

## Repetition policy

Repeat the Tier 1 conditions with additional seeds when compute allows. Report mean and dispersion rather than a single-run number when repeated runs exist.

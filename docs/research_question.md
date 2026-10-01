# Research Question

## Research Question

Can curriculum-guided fine-tuning combined with response-level teacher supervision improve code-reasoning performance in a small language model compared with matched single-stage supervised fine-tuning and with each component evaluated separately?

## Hypotheses

- H1: ordered curriculum improves performance over matched mixed-data SFT.
- H2: teacher-generated supervision improves performance over ordinary SFT.
- H3: curriculum and teacher supervision interact and should be analyzed jointly.
- H4: stage ordering affects final capability.
- H5: any efficiency advantage must be demonstrated by direct measurement.

All are hypotheses until measured.

## Variables

Independent: curriculum mode, teacher supervision on/off, teacher identity/diversity, training budget.

Dependent: solve rate, compile rate, execution pass rate, pass@k, capability-specific scores, latency and throughput.

Controls: base student, dataset revisions when available, prompt template, sequence length, optimizer family, seed, hardware profile.

## Threats to Validity

Contamination, overlap, unequal token/step budgets, teacher version drift, prompt sensitivity, small evaluation sets, and execution-environment variance.

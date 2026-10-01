# Legacy Results Record

This file preserves the historical results section from the original README for provenance.
These numbers are **legacy / unverified** and are not treated as scientific results by the
CGKD-CR framework because the original repository did not include a reproducible benchmark
manifest, held-out split, seed record, or machine-readable evaluation artifact tying these
values to a specific run.

## Historical sample table

| Metric | Before training | After Stage 2 | After Stage 4 (+Distill) |
|---|---:|---:|---:|
| Simple algorithmic accuracy (IOI-style, small sample) | ~38% | ~57% | ~66% |
| Output format compliance (Pass@Format) | ~62% | ~81% | ~89% |
| Common syntax/runtime errors | High | Medium | Low |

The legacy README itself described these as indicative values from lightweight internal
checks over small samples.

## Historical qualitative observations

The original README reported:

- clearer instruction following and better output-format adherence
- more coherent step-by-step reasoning on algorithmic tasks
- fewer common debugging mistakes and clearer fixes
- broader coverage of complex coding instructions

It also contained example-level before/after descriptions for an LCS task, debugging
edge cases, and structured JSON output.

## Research treatment

These observations are retained here only as historical provenance. They must not be
copied into current results tables, abstracts, conclusions, or claims about CGKD-CR until
the same or a stronger evaluation protocol reproduces them.

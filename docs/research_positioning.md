# Research Positioning

## Known techniques

SFT, LoRA/PEFT, curriculum learning, and teacher-student response generation are established techniques.

## Engineering integration

The original repository integrated these components in a Colab notebook. The refactor separates configuration, data, teacher supervision, training, checkpoint lineage, evaluation, and result serialization.

## Research question

The study tests separate and combined effects under controlled budgets.

## Potential contribution

Any novelty claim is conditional on empirical evidence. The repository does not treat the combination of known techniques as automatically novel.

## Terminology

The present teacher path is response-level supervision: teacher outputs become student targets. It is not logit matching or hidden-state distillation.

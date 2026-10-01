# Reproducibility

The evaluator uses the same base model, prompt template, decoding settings and test suite for every checkpoint.

Example:

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_checkpoints.py --base-model unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit --checkpoint base
python scripts/evaluate_checkpoints.py --base-model unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit --checkpoint stage1-foundation
python scripts/evaluate_checkpoints.py --base-model unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit --checkpoint stage2-algorithms
python scripts/evaluate_checkpoints.py --base-model unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit --checkpoint stage3-debugging
python scripts/evaluate_checkpoints.py --base-model unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit --checkpoint stage4-advanced
```

The `--checkpoint` path points to a LoRA adapter directory. `base` evaluates the frozen student without an adapter.

For the causal curriculum-vs-pooled comparison, keep total training examples/tokens, optimizer family, LR schedule and LoRA target modules matched.
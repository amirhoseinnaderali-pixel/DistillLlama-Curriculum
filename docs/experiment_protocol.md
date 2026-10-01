# Experiment Protocol

1. Record git commit and config.
2. Freeze the seed.
3. Fingerprint train/eval datasets.
4. Keep manifests immutable for the run.
5. Run the four central matrix conditions.
6. Repeat key conditions across additional seeds where possible.
7. Evaluate on unseen items with execution-based metrics.
8. Serialize results with results/schema.json.
9. Perform capability-transfer and error analysis.
10. Never replace Pending evaluation with a number not produced by a recorded run.

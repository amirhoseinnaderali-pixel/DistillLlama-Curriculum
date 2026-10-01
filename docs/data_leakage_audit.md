# Data Leakage Audit

Status: Not yet fully audited.

Required checks:
1. exact or normalized train-eval overlap
2. duplicate problems within and across stages
3. repeated IOI items
4. teacher outputs that reveal evaluation answers
5. benchmark contamination
6. preprocessing or formatting label leakage

A clean-evaluation claim requires the overlap report to be stored with experiment artifacts.

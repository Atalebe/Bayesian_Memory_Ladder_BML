# Reproducibility

This repository is a curated public snapshot of the Bayesian Memory Ladder (BML) branch used in the PRD reply workflow.

## Included branches
Included public branches:
- 0111: exploratory full-memory MCMC
- 0112: baseline LCDM MCMC
- 0114: restricted-memory MCMC
- 0115: baseline LCDM PolyChord evidence
- 0116: restricted-memory PolyChord evidence
- 0118: controlled full-memory PolyChord evidence

## Excluded branches
See `docs/EXCLUDED_RUNS.md`.

## What is reproduced here
This snapshot is intended to reproduce:
- the evidence ledger
- the reply figures
- the restricted-memory reject audit
- the posterior summaries used in the reply

## Regenerate reply figures
From repository root:
```bash
python scripts/reply/01_build_evidence_ledger.py
python scripts/reply/02_plot_evidence_comparison.py
python scripts/reply/03_plot_kstar_posterior.py
python scripts/reply/04_plot_numerical_audit.py
python scripts/reply/05_plot_h0_comparison.py

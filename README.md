# Bayesian Memory Ladder

Public reproducibility snapshot for the Bayesian Memory Ladder (BML) primordial-memory branch used in the PRD reply workflow.

## What this repository contains

This repository contains the code, sanitized configurations, curated run artifacts, reply-figure scripts, and frozen evidence snapshots needed to inspect the validated BML branches used in the manuscript.

Included public branches:
- 0111: exploratory full-memory MCMC
- 0112: baseline LCDM MCMC
- 0114: restricted-memory MCMC
- 0115: baseline LCDM PolyChord evidence
- 0116: restricted-memory PolyChord evidence
- 0118: controlled full-memory PolyChord evidence

## What is excluded

Miscentered or scientifically uninterpretable branches are not included in the public artifact set.

See `docs/EXCLUDED_RUNS.md`.

## Repository layout

- `src/bml/`: core BML implementation
- `configs/`: sanitized public run configurations
- `artifacts/`: curated run outputs used in the manuscript
- `scripts/reply/`: scripts that regenerate the reply tables and figures
- `outputs/reply/`: generated reply figures and tables
- `results/snapshots/`: frozen evidence ledgers for public reference
- `docs/`: reproducibility and exclusion notes

## Regenerating the reply figures

From repository root:

```bash
./scripts/reply/build_reply_bundle.sh

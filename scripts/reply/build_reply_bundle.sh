#!/usr/bin/env bash
set -euo pipefail
python scripts/reply/01_build_evidence_ledger.py
python scripts/reply/02_plot_evidence_comparison.py
python scripts/reply/03_plot_kstar_posterior.py
python scripts/reply/04_plot_numerical_audit.py
python scripts/reply/05_plot_h0_comparison.py

# Reply figure scripts, revised

This revision fixes three issues from the first bundle:

1. `01_build_evidence_ledger.py` no longer writes undeclared dict fields.
2. `02_plot_evidence_comparison.py` reads the rebuilt ledger and plots real `Δlog Z` values.
3. `03_plot_kstar_posterior.py` now parses PolyChord equal-weight files correctly by using:
   - the first 2 columns as bookkeeping,
   - the next `nDims` columns as sampled parameters,
   - `k_*` from the known restricted-memory parameter ordering.

The scripts are still designed for active, frozen, or converged snapshots.

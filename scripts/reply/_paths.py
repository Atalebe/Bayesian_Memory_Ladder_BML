from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
OUT = ROOT / "outputs" / "reply"
FIG = OUT / "figures"
TAB = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

RUN = {
    "0115": {
        "stats": ART / "0115" / "run_0115_planck_polychord_LCDM_evidence.stats",
        "equal_weights": ART / "0115" / "run_0115_planck_polychord_LCDM_evidence_equal_weights.txt",
    },
    "0116": {
        "stats": ART / "0116" / "run_0116_planck_polychord_memory_restricted_v2_evidence.stats",
        "equal_weights": ART / "0116" / "run_0116_planck_polychord_memory_restricted_v2_evidence_equal_weights.txt",
        "diag_gz": ART / "0116" / "run_0116_bml_theory_diag_snapshot_2026-04-27.jsonl.gz",
        "reject_audit": ART / "0116" / "reject_audit_0116_snapshot_2026-04-27.json",
    },
    "0118": {
        "stats": ART / "0118" / "run_0118_planck_polychord_memory_full_controlled_evidence.stats",
        "equal_weights": ART / "0118" / "run_0118_planck_polychord_memory_full_controlled_evidence_equal_weights.txt",
    },
}

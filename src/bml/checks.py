from __future__ import annotations
from pathlib import Path
import numpy as np

def compare_pk_tables(pk_a: Path, pk_b: Path, rtol: float = 1e-4) -> dict:
    a = np.loadtxt(pk_a)
    b = np.loadtxt(pk_b)

    ka, pa = a[:, 0], a[:, 1]
    kb, pb = b[:, 0], b[:, 1]

    if not np.allclose(ka, kb, rtol=0, atol=0):
        pb_i = np.interp(ka, kb, pb)
    else:
        pb_i = pb

    rel = np.abs(pa - pb_i) / np.maximum(np.abs(pa), 1e-60)
    max_rel = float(np.max(rel))
    ok = bool(max_rel <= rtol)

    return {"ok": ok, "rtol": rtol, "max_rel": max_rel, "pk_a": str(pk_a), "pk_b": str(pk_b)}

from __future__ import annotations
from pathlib import Path
import numpy as np

def class_available() -> bool:
    try:
        from classy import Class  # noqa: F401
        return True
    except Exception:
        return False

def run_class_powerlaw(As: float, ns: float, pivot_k: float, ell_max: int, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    from classy import Class

    cosmo = Class()
    params = {
        "output": "tCl,pCl,lCl",
        "l_max_scalars": int(ell_max),
        "h": 0.674,
        "omega_b": 0.0224,
        "omega_cdm": 0.120,
        "A_s": As,
        "n_s": ns,
        "k_pivot": pivot_k,
        "tau_reio": 0.054,
    }
    cosmo.set(params)
    cosmo.compute()

    cls = cosmo.lensed_cl(ell_max)
    saved = {}
    for key, arr in cls.items():
        p = outdir / f"class_{key}.txt"
        np.savetxt(p, arr, header=f"{key} lensed_cl, row index is ell")
        saved[key] = str(p)

    cosmo.struct_cleanup()
    cosmo.empty()
    return {"ok": True, "saved": saved, "ell_max": ell_max}

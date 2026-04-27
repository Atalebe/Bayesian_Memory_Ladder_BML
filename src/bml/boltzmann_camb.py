from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def camb_available() -> bool:
    try:
        import camb  # noqa: F401
        return True
    except Exception:
        return False


def _load_pk_table(pk_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(pk_path)
    k = d[:, 0].astype(float)
    pk = d[:, 1].astype(float)
    order = np.argsort(k)
    return k[order], pk[order]


def _ns_eff_from_table(k: np.ndarray, pk: np.ndarray, kpiv: float) -> float:
    # P_R(k) ~ k^(ns-1) => d ln P / d ln k = ns-1
    logk = np.log(k)
    logp = np.log(np.maximum(pk, 1e-300))

    i = int(np.searchsorted(k, kpiv))
    i = max(1, min(i, len(k) - 2))

    slope = (logp[i + 1] - logp[i - 1]) / (logk[i + 1] - logk[i - 1])  # = ns-1
    return float(slope + 1.0)


def run_camb_from_pk_table(pk_path: str | Path, out_dir: str | Path, camb_cfg: Dict, cosmo_cfg: Dict) -> dict:
    import camb
    from camb import model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k, pk = _load_pk_table(Path(pk_path))

    # cosmology (Planck-ish defaults; override in config)
    H0 = float(cosmo_cfg.get("H0", 67.66))
    ombh2 = float(cosmo_cfg.get("ombh2", 0.02237))
    omch2 = float(cosmo_cfg.get("omch2", 0.1200))
    tau = float(cosmo_cfg.get("tau", 0.0544))
    mnu = float(cosmo_cfg.get("mnu", 0.06))
    omk = float(cosmo_cfg.get("omk", 0.0))

    lmax = int(camb_cfg.get("lmax", 2500))
    lens_potential_accuracy = int(camb_cfg.get("lens_potential_accuracy", 1))

    # pivot used only for ns_eff inference
    kpiv = float(cosmo_cfg.get("kpiv", camb_cfg.get("kpiv", 0.05)))
    ns_eff = _ns_eff_from_table(k, pk, kpiv)

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=lens_potential_accuracy)

    # Make absolutely sure CAMB doesn't try to do nonlinear corrections
    pars.NonLinear = model.NonLinear_none
    try:
        pars.NonLinearLens = False
    except Exception:
        pass

    # Primordial: tabulated P_R(k)
    pars.set_initial_power_table(k, pk)

    # Critical: provide effective tilt for the splined power
    try:
        pars.InitPower.effective_ns_for_nonlinear = ns_eff
    except Exception:
        # older/newer CAMB builds differ; if attribute doesn't exist, no harm
        pass

    results = camb.get_results(pars)
    spectra = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    total = spectra["total"]  # ell x [TT, EE, BB, TE]
    ell = np.arange(total.shape[0])

    out = out_dir / "camb_cls_total.dat"
    np.savetxt(out, np.column_stack([ell, total]), header="ell TT EE BB TE")

    return {"cls_total": str(out), "lmax": lmax, "ns_eff": ns_eff}

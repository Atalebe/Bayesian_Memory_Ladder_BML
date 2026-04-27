from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
import camb

import bml.cobaya_bml_theory as bml_mod


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _fractional_diff(a: np.ndarray, b: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(bb), floor)
    return np.abs(aa - bb) / denom


def _interp_to_ref(x_src: np.ndarray, y_src: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    return np.interp(x_ref, x_src, y_src)


def _make_theory(theory_cfg: dict):
    th = bml_mod.bml_theory()
    for k, v in theory_cfg.items():
        setattr(th, k, v)
    th.initialize()
    return th


def _compute_memory(theory_cfg: dict, fixed_point: dict, point: dict):
    th = _make_theory(theory_cfg)

    state = {}
    p = {
        "As": fixed_point["As"],
        "ns": fixed_point["ns"],
        "alpha": point["alpha"],
        "k_star": point["k_star"],
        "sigma_ln": point["sigma_ln"],
    }

    t0 = perf_counter()
    ok = th.calculate(state, want_derived=True, **p)
    dt = perf_counter() - t0

    out = {
        "memory_ok": bool(ok),
        "memory_wall_sec": dt,
    }

    if not ok:
        return out

    pk = state["primordial_scalar_pk"]
    out["pk"] = {
        "k": np.asarray(pk["k"], dtype=float),
        "Pk": np.asarray(pk["Pk"], dtype=float),
        "kmin": float(pk["kmin"]),
        "kmax": float(pk["kmax"]),
        "pmin": float(np.min(pk["Pk"])),
        "pmax": float(np.max(pk["Pk"])),
        "finite": bool(np.isfinite(pk["Pk"]).all()),
        "positive": bool((np.asarray(pk["Pk"]) > 0).all()),
    }
    out["derived"] = state.get("derived", {})
    return out


def _get_camb_nonlinear_mode(mode_name: str):
    mode_name = str(mode_name).strip().lower()
    if mode_name in {"none", "nonlinear_none", "off", "false"}:
        return camb.model.NonLinear_none
    if mode_name in {"lens", "nonlinear_lens", "lensing"}:
        return camb.model.NonLinear_lens
    if mode_name in {"both", "nonlinear_both"}:
        return camb.model.NonLinear_both
    raise ValueError(f"Unknown CAMB nonlinear mode: {mode_name}")


def _compute_camb(camb_cfg: dict, fixed_point: dict, pk: dict):
    cp = camb.CAMBparams()
    cp.set_cosmology(
        H0=fixed_point["H0"],
        ombh2=fixed_point["ombh2"],
        omch2=fixed_point["omch2"],
        tau=fixed_point["tau"],
        mnu=fixed_point["mnu"],
        omk=fixed_point["omk"],
    )

    cp.WantCls = True
    cp.Want_CMB = True
    cp.WantTransfer = True

    nonlinear_mode = camb_cfg.get("nonlinear_mode", "none")
    cp.NonLinear = _get_camb_nonlinear_mode(nonlinear_mode)

    lmax = int(camb_cfg["extra_args"]["lmax"])
    lens_potential_accuracy = int(camb_cfg["extra_args"]["lens_potential_accuracy"])
    cp.set_for_lmax(lmax=lmax, lens_potential_accuracy=lens_potential_accuracy)

    ip = camb.initialpower.SplinedInitialPower()
    ip.set_scalar_table(pk["k"], pk["Pk"])

    # This is the critical fix for splined primordial spectra.
    ip.effective_ns_for_nonlinear = float(fixed_point["ns"])

    cp.set_initial_power(ip)

    t0 = perf_counter()
    results = camb.get_results(cp)
    cl = results.get_cmb_power_spectra(raw_cl=False)["total"]
    dt = perf_counter() - t0

    out = {
        "camb_ok": bool(np.isfinite(cl).all()),
        "camb_wall_sec": dt,
        "cl_total": cl,
        "tt_200": float(cl[200, 0]) if cl.shape[0] > 200 else np.nan,
        "tt_1000": float(cl[1000, 0]) if cl.shape[0] > 1000 else np.nan,
        "tt_2000": float(cl[2000, 0]) if cl.shape[0] > 2000 else np.nan,
    }
    return out


def _max_log_slope_abs(k: np.ndarray, pk: np.ndarray) -> float:
    lk = np.log(np.asarray(k, dtype=float))
    lp = np.log(np.asarray(pk, dtype=float))
    dlp = np.diff(lp)
    dlk = np.diff(lk)
    slope = dlp / dlk
    return float(np.max(np.abs(slope)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="benchmark YAML")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    out_jsonl = Path(cfg["output_jsonl"])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    fixed_point = cfg["fixed_point"]
    camb_cfg = cfg["theory"]["camb"]
    base_theory_cfg = cfg["theory"]["bml.cobaya_bml_theory.bml_theory"]

    ref_cfg = cfg["reference"]
    ref_point = cfg["benchmark_points"][0]

    ref_theory_cfg = copy.deepcopy(base_theory_cfg)
    ref_theory_cfg["nk"] = int(ref_cfg["nk"])
    ref_theory_cfg["method"] = ref_cfg["method"]
    ref_theory_cfg["rtol"] = float(ref_cfg["rtol"])
    ref_theory_cfg["atol"] = float(ref_cfg["atol"])

    print("Building reference solution...")
    ref_mem = _compute_memory(ref_theory_cfg, fixed_point, ref_point)
    if not ref_mem["memory_ok"]:
        raise RuntimeError("Reference memory solve failed")

    ref_camb = _compute_camb(camb_cfg, fixed_point, ref_mem["pk"])
    if not ref_camb["camb_ok"]:
        raise RuntimeError("Reference CAMB solve failed")

    ref_pk = ref_mem["pk"]
    ref_cl = ref_camb["cl_total"]

    methods = cfg["scan"]["method"]
    nks = cfg["scan"]["nk"]
    rtols = cfg["scan"]["rtol"]
    atols = cfg["scan"]["atol"]

    rows = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for point in cfg["benchmark_points"]:
            for method, nk, rtol, atol in itertools.product(methods, nks, rtols, atols):
                theory_cfg = copy.deepcopy(base_theory_cfg)
                theory_cfg["method"] = method
                theory_cfg["nk"] = int(nk)
                theory_cfg["rtol"] = float(rtol)
                theory_cfg["atol"] = float(atol)

                row = {
                    "point_name": point["name"],
                    "alpha": float(point["alpha"]),
                    "k_star": float(point["k_star"]),
                    "sigma_ln": float(point["sigma_ln"]),
                    "method": str(method),
                    "nk": int(nk),
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "memory_ok": False,
                    "camb_ok": False,
                    "total_ok": False,
                }

                try:
                    mem = _compute_memory(theory_cfg, fixed_point, point)
                    row["memory_ok"] = bool(mem["memory_ok"])
                    row["memory_wall_sec"] = _safe_float(mem["memory_wall_sec"])

                    if row["memory_ok"]:
                        pk = mem["pk"]
                        row["pk_kmin"] = pk["kmin"]
                        row["pk_kmax"] = pk["kmax"]
                        row["pk_pmin"] = pk["pmin"]
                        row["pk_pmax"] = pk["pmax"]
                        row["pk_finite"] = pk["finite"]
                        row["pk_positive"] = pk["positive"]
                        row["max_log_slope_abs"] = _max_log_slope_abs(pk["k"], pk["Pk"])

                        derived = mem.get("derived", {})
                        for key, val in derived.items():
                            row[f"derived_{key}"] = _safe_float(val)

                        pk_interp = _interp_to_ref(pk["k"], pk["Pk"], ref_pk["k"])
                        frac_pk = _fractional_diff(pk_interp, ref_pk["Pk"])
                        row["pk_fracdiff_max_vs_ref"] = float(np.max(frac_pk))
                        row["pk_fracdiff_p95_vs_ref"] = float(np.percentile(frac_pk, 95.0))

                        camb_out = _compute_camb(camb_cfg, fixed_point, pk)
                        row["camb_ok"] = bool(camb_out["camb_ok"])
                        row["camb_wall_sec"] = _safe_float(camb_out["camb_wall_sec"])

                        if row["camb_ok"]:
                            row["tt_200"] = camb_out["tt_200"]
                            row["tt_1000"] = camb_out["tt_1000"]
                            row["tt_2000"] = camb_out["tt_2000"]

                            test_cl = camb_out["cl_total"]
                            max_ell = min(test_cl.shape[0], ref_cl.shape[0]) - 1
                            ell_samples = [200, 1000, 2000]
                            for ell in ell_samples:
                                if ell <= max_ell:
                                    for col, label in [(0, "TT"), (1, "EE"), (3, "TE")]:
                                        ref_val = ref_cl[ell, col]
                                        test_val = test_cl[ell, col]
                                        frac = abs(test_val - ref_val) / max(abs(ref_val), 1e-300)
                                        row[f"{label}_{ell}_fracdiff_vs_ref"] = float(frac)

                            row["total_ok"] = True
                            row["total_wall_sec"] = float(
                                row["memory_wall_sec"] + row["camb_wall_sec"]
                            )

                except Exception as e:
                    row["exception"] = repr(e)

                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                print(row)
                rows.append(row)

    good = [r for r in rows if r.get("total_ok", False)]
    print()
    print(f"total rows: {len(rows)}")
    print(f"good rows : {len(good)}")

    if good:
        best = sorted(good, key=lambda r: (r["total_wall_sec"], r["pk_fracdiff_max_vs_ref"]))[:10]
        print("\nTop 10 fastest good configurations:")
        for r in best:
            print(
                r["point_name"],
                r["method"],
                r["nk"],
                r["rtol"],
                r["atol"],
                "total_wall_sec=",
                round(r["total_wall_sec"], 3),
                "pk_fracdiff_max_vs_ref=",
                "{:.3e}".format(r["pk_fracdiff_max_vs_ref"]),
            )


if __name__ == "__main__":
    main()

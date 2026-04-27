from __future__ import annotations

import sys
import shutil
import traceback
from pathlib import Path

import numpy as np

from .config import load_config
from .logging_utils import system_info, write_json, utc_now
from .primordial import make_k_grid, power_law_pk, save_pk_table
from .plotting import plot_pk
from .modecode_adapter import run_modecode
from .checks import compare_pk_tables

from .boltzmann_camb import camb_available, run_camb_from_pk_table
from .boltzmann_class import class_available, run_class_powerlaw

from .memory_solver import solve_pk_memory
from .memory_u_solver import solve_pk_memory_u


def _plot_pk_auto(pk_path: Path, out_path: Path, run_id: str) -> None:
    # Support both historical signatures:
    #   plot_pk(pk_path, out_path)  OR  plot_pk(k, pk, out_path, title=...)
    try:
        plot_pk(pk_path, out_path)
        return
    except TypeError:
        pass

    dat = np.loadtxt(pk_path)
    k, pk = dat[:, 0], dat[:, 1]
    plot_pk(k, pk, out_path, title=f"Run {run_id}: P_R(k)")


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    out_root: Path = cfg.output_root

    logs_dir = out_root / "logs"
    pk_dir = out_root / "pk"
    fig_dir = out_root / "figures"
    cls_dir = out_root / "cls"
    tmp_dir = out_root / "tmp"

    for d in (logs_dir, pk_dir, fig_dir, cls_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # snapshot config
    cfg_snapshot = logs_dir / "config_snapshot.yml"
    shutil.copyfile(cfg_path, cfg_snapshot)

    manifest = {
        "run_id": cfg.run_id,
        "config_path": str(Path(cfg_path).resolve()),
        "output_root": str(out_root),
        "timestamp_utc": utc_now(),
        "system": system_info(),
        "steps": {},
    }

    try:
        pcfg = cfg.raw.get("primordial", {})
        model = str(pcfg.get("model", "power_law")).lower()

        pk_path = pk_dir / f"pk_run_{cfg.run_id}.txt"

        # ---------------- Primordial ----------------
        if model == "power_law":
            k = make_k_grid(cfg.raw)
            pk = power_law_pk(k, cfg.raw)
            save_pk_table(pk_path, k, pk, header="k_Mpc^-1    P_R(k)    (power-law)")
            manifest["steps"]["primordial"] = {"ok": True, "nk": int(len(k)), "pk_table": str(pk_path)}

        elif model == "from_modecode":
            pk_path = run_modecode(cfg.raw, run_tmp=tmp_dir / "modecode", logs_dir=logs_dir, pk_dir=pk_dir)
            manifest["steps"]["modecode"] = {"ok": True, "pk_table": str(pk_path)}

        elif model == "memory_solver":
            diag_path = pk_dir / f"freeze_diag_run_{cfg.run_id}.txt"
            diag = solve_pk_memory(cfg.raw, out_pk_path=pk_path, diagnostics_path=diag_path)
            manifest["steps"]["memory_solver"] = {"ok": True, "pk_table": str(pk_path), "freeze_diag": str(diag_path), "diag": diag}

        elif model == "memory_u_solver":
            diag_path = pk_dir / f"freeze_diag_run_{cfg.run_id}.txt"
            diag = solve_pk_memory_u(cfg.raw, out_pk_path=pk_path, diagnostics_path=diag_path)
            manifest["steps"]["memory_u_solver"] = {"ok": True, "pk_table": str(pk_path), "freeze_diag": str(diag_path), "diag": diag}

        else:
            raise ValueError(f"Unknown primordial model: {model}")

        # ---------------- Plot ----------------
        pk_plot_path = fig_dir / f"pk_run_{cfg.run_id}.png"
        _plot_pk_auto(pk_path, pk_plot_path, cfg.run_id)
        manifest["steps"]["pk_plot"] = {"ok": True, "pk_plot": str(pk_plot_path)}

        # ---------------- Checks ----------------
        checks_cfg = cfg.raw.get("checks", {})
        if bool(checks_cfg.get("enabled", False)):
            ml = checks_cfg.get("markov_limit", None)
            if ml:
                ref = Path(ml["compare_to_run"])
                rtol = float(ml.get("rtol", 1e-4))
                ck = compare_pk_tables(pk_path, ref, rtol=rtol)
                manifest["steps"]["check_markov_limit"] = ck
                if not ck.get("ok", False):
                    raise RuntimeError(f"Markov limit check failed (max_rel={ck['max_rel']:.3e} > rtol={rtol})")

        # ---------------- CAMB (Phase III) ----------------
        ccfg = cfg.raw.get("camb", {})
        if bool(ccfg.get("enabled", False)):
            if camb_available():
                cosmo = cfg.raw.get("cosmo", {})
                camb_out = run_camb_from_pk_table(pk_path, out_dir=cls_dir / "camb", camb_cfg=ccfg, cosmo_cfg=cosmo)
                manifest["steps"]["camb"] = {"ok": True, **camb_out}
            else:
                manifest["steps"]["camb"] = {"ok": False, "skipped": True, "reason": "CAMB not available"}
        else:
            manifest["steps"]["camb"] = {"ok": False, "skipped": True, "reason": "CAMB disabled"}

        # ---------------- CLASS (optional; power-law only here) ----------------
        class_cfg = cfg.raw.get("class", {"enabled": False})
        if bool(class_cfg.get("enabled", False)) and class_available():
            if model != "power_law":
                manifest["steps"]["class"] = {"ok": False, "skipped": True, "reason": "CLASS external P(k) ingest not wired yet"}
            else:
                manifest["steps"]["class"] = run_class_powerlaw(
                    As=float(pcfg["As"]),
                    ns=float(pcfg["ns"]),
                    pivot_k=float(pcfg["pivot_k"]),
                    ell_max=int(class_cfg.get("ell_max", 2500)),
                    outdir=cls_dir / "class",
                )
        else:
            manifest["steps"]["class"] = {"ok": False, "skipped": True, "reason": "CLASS disabled or not installed"}

        manifest["ok"] = True

    except Exception as e:
        manifest["ok"] = False
        manifest["error"] = repr(e)
        manifest["traceback"] = traceback.format_exc()

    write_json(logs_dir / "run_manifest.json", manifest)

    if not manifest["ok"]:
        print("Run failed, see logs/run_manifest.json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m bml.run <config.yml>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])

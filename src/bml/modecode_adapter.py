from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

def _run(cmd: list[str], cwd: Path, logfile: Path, env: dict[str, str] | None = None) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}. See {logfile}")

def _detect_build(root: Path) -> list[str] | None:
    if (root / "Makefile").exists() or (root / "makefile").exists():
        return ["make", "-j"]
    if (root / "CMakeLists.txt").exists():
        return ["cmake", "-S", ".", "-B", "build"]
    return None

def run_modecode(cfg: dict, run_tmp: Path, logs_dir: Path, pk_dir: Path) -> Path:
    mc = cfg.get("modecode", {})
    enabled = bool(mc.get("enabled", False))
    if not enabled:
        raise ValueError("ModeCode not enabled")

    root = Path(mc.get("root", "external/modecode")).resolve()
    exe = root / mc["executable"]
    param_file = root / mc["param_file"]
    pk_file = Path(mc["pk_file"])

    if not root.exists():
        raise FileNotFoundError(f"ModeCode root not found: {root}")
    if not exe.exists():
        raise FileNotFoundError(f"ModeCode executable not found: {exe}")
    if not param_file.exists():
        raise FileNotFoundError(f"ModeCode param file not found: {param_file}")

    # Optional build
    build = mc.get("build", {})
    if bool(build.get("enabled", False)):
        cmd = build.get("cmd", None) or _detect_build(root)
        if cmd is None:
            raise RuntimeError("build.enabled=true but no build system detected; set modecode.build.cmd")
        _run(cmd, cwd=root, logfile=logs_dir / "modecode_build.log")

    # Work dir: keep outputs inside the run folder
    run_tmp.mkdir(parents=True, exist_ok=True)
    tmp_param = run_tmp / "modecode_params.ini"
    shutil.copyfile(param_file, tmp_param)

    # Convenience: pass memory params as environment vars (your fork can read them)
    env = os.environ.copy()
    mem = cfg.get("memory", {})
    if bool(mem.get("enabled", False)):
        env["BML_MEMORY_ENABLED"] = "1"
        env["BML_ALPHA"] = str(mem.get("alpha", 0.0))
        env["BML_BETA"] = str(mem.get("beta", 0.0))
    else:
        env["BML_MEMORY_ENABLED"] = "0"
        env["BML_ALPHA"] = "0.0"
        env["BML_BETA"] = "0.0"

    _run([str(exe), str(tmp_param)], cwd=run_tmp, logfile=logs_dir / "modecode_run.log", env=env)

    pk_path = pk_file if pk_file.is_absolute() else (run_tmp / pk_file)
    if not pk_path.exists():
        candidates = list(run_tmp.rglob("Pscalar*.dat")) + list(run_tmp.rglob("*pk*scalar*.dat"))
        if candidates:
            pk_path = candidates[0]
        else:
            raise FileNotFoundError(f"ModeCode pk file not found: {pk_path}")

    data = np.loadtxt(pk_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected pk format in {pk_path}: shape {data.shape}")

    k = data[:, 0]
    pr = data[:, 1]

    out = pk_dir / f"pk_run_{cfg['run_id']}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "k_Mpc^-1    P_R(k)    (exported from ModeCode)"
    np.savetxt(out, np.column_stack([k, pr]), header=header)
    return out

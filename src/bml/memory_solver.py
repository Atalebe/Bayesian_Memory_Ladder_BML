from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class MemorySolverConfig:
    kmin: float
    kmax: float
    nk: int
    spacing: str

    alpha: float
    beta: float

    pivot_k: float
    As: float
    renormalize_to_As: bool = True

    x_start: float = 80.0
    x_end: float = 1e-3
    method: str = "BDF"
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float | None = None


def _k_grid(kmin: float, kmax: float, nk: int, spacing: str) -> np.ndarray:
    if spacing.lower() == "log":
        return np.logspace(np.log10(kmin), np.log10(kmax), nk)
    if spacing.lower() == "linear":
        return np.linspace(kmin, kmax, nk)
    raise ValueError(f"Unknown spacing: {spacing}")


def _bd_u_eta(k: float, eta: float) -> Tuple[complex, complex]:
    u = np.exp(-1j * k * eta) / np.sqrt(2.0 * k)
    up = (-1j * k) * u
    return u, up


def _rhs_x(x: float, y: np.ndarray, alpha: float, beta: float, memory_enabled: bool) -> np.ndarray:
    R = y[0] + 1j * y[1]
    P = y[2] + 1j * y[3]
    Y = y[4] + 1j * y[5]

    Rx = P

    if x == 0.0:
        Px = -R + (alpha * Y if memory_enabled else 0.0)
    else:
        Px = (2.0 / x) * P - R + (alpha * Y if memory_enabled else 0.0)

    Yx = (beta * Y + P) if memory_enabled else (0.0 + 0.0j)

    return np.array([Rx.real, Rx.imag, Px.real, Px.imag, Yx.real, Yx.imag], dtype=float)


def solve_pk_memory(cfg: Dict, out_pk_path: Path, diagnostics_path: Path | None = None) -> dict:
    kcfg = cfg["k_grid"]
    mem = cfg["memory"]
    pcfg = cfg.get("primordial", {})
    icfg = cfg.get("integration", {})

    ms = MemorySolverConfig(
        kmin=float(kcfg["kmin"]),
        kmax=float(kcfg["kmax"]),
        nk=int(kcfg["nk"]),
        spacing=str(kcfg.get("spacing", "log")),
        alpha=float(mem.get("alpha", 0.0)),
        beta=float(mem.get("beta", 0.0)),
        pivot_k=float(pcfg.get("pivot_k", 0.05)),
        As=float(pcfg.get("As", 2.1e-9)),
        renormalize_to_As=bool(pcfg.get("renormalize_to_As", True)),
        x_start=float(icfg.get("x_start", 80.0)),
        x_end=float(icfg.get("x_end", 1e-3)),
        method=str(icfg.get("method", "BDF")),
        rtol=float(icfg.get("rtol", 1e-8)),
        atol=float(icfg.get("atol", 1e-10)),
        max_step=None if icfg.get("max_step", None) is None else float(icfg["max_step"]),
    )

    if not (ms.x_start > ms.x_end > 0.0):
        raise ValueError(f"Require x_start > x_end > 0. Got x_start={ms.x_start}, x_end={ms.x_end}")

    memory_enabled = bool(mem.get("enabled", False)) and (abs(ms.alpha) > 0.0)

    k_arr = _k_grid(ms.kmin, ms.kmax, ms.nk, ms.spacing)
    P_raw = np.zeros_like(k_arr)
    freeze_ratio = np.zeros_like(k_arr)

    s_end = ms.x_start - ms.x_end

    ivp_opts = dict(method=ms.method, rtol=ms.rtol, atol=ms.atol)
    if ms.max_step is not None:
        ivp_opts["max_step"] = ms.max_step

    for i, k in enumerate(k_arr):
        x0 = ms.x_start
        eta0 = -x0 / k

        z0 = k / x0
        zp0 = (k * k) / (x0 * x0)

        u0, up0 = _bd_u_eta(k, eta0)

        R0 = u0 / z0
        Rprime0 = (up0 / z0) - (zp0 / (z0 * z0)) * u0
        P0 = -Rprime0 / k

        Y0 = 0.0 + 0.0j

        y0 = np.array([R0.real, R0.imag, P0.real, P0.imag, Y0.real, Y0.imag], dtype=float)

        def rhs_s(s: float, yy: np.ndarray) -> np.ndarray:
            x = ms.x_start - s
            return -_rhs_x(x, yy, alpha=ms.alpha, beta=ms.beta, memory_enabled=memory_enabled)

        sol = solve_ivp(fun=rhs_s, t_span=(0.0, s_end), y0=y0, **ivp_opts)
        if not sol.success:
            raise RuntimeError(f"Memory ODE failed for k={k:.3e}: {sol.message}")

        y_end = sol.y[:, -1]
        R_end = y_end[0] + 1j * y_end[1]
        P_end = y_end[2] + 1j * y_end[3]

        P_raw[i] = (k**3 / (2.0 * np.pi**2)) * (np.abs(R_end) ** 2)
        freeze_ratio[i] = float((k * np.abs(P_end)) / max(np.abs(R_end), 1e-60))

    renorm = 1.0
    if ms.renormalize_to_As:
        if not (k_arr.min() <= ms.pivot_k <= k_arr.max()):
            raise ValueError(f"pivot_k={ms.pivot_k} outside k grid [{k_arr.min()}, {k_arr.max()}]")
        P_piv = float(np.interp(ms.pivot_k, k_arr, P_raw))
        renorm = ms.As / P_piv if P_piv > 0 else 1.0

    P = renorm * P_raw

    out_pk_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_pk_path,
        np.column_stack([k_arr, P]),
        header="k_Mpc^-1    P_R(k)    (memory_solver, forward s=x_start-x integration, renorm to As if enabled)",
    )

    diag = {
        "alpha": ms.alpha,
        "beta": ms.beta,
        "memory_enabled": memory_enabled,
        "x_start": ms.x_start,
        "x_end": ms.x_end,
        "method": ms.method,
        "rtol": ms.rtol,
        "atol": ms.atol,
        "max_step": ms.max_step,
        "renormalize_to_As": ms.renormalize_to_As,
        "pivot_k": ms.pivot_k,
        "As": ms.As,
        "renorm_factor": renorm,
        "freeze_ratio_max": float(np.max(freeze_ratio)),
        "freeze_ratio_p95": float(np.percentile(freeze_ratio, 95)),
    }

    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            diagnostics_path,
            np.column_stack([k_arr, freeze_ratio]),
            header="k_Mpc^-1    freeze_ratio=|R'|/|R| at x_end",
        )

    return diag

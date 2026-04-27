from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap


class MemorySolverFailure(RuntimeError):
    """Numerical or runtime failure for a proposed point. Caller should reject cleanly."""
    pass


def _append_jsonl(path: Optional[str | Path], row: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _k_grid(kmin: float, kmax: float, nk: int, spacing: str) -> np.ndarray:
    if nk < 2:
        raise ValueError(f"nk must be >= 2, got {nk}")
    if kmin <= 0.0 or kmax <= 0.0:
        raise ValueError(f"kmin and kmax must be > 0, got {kmin}, {kmax}")
    if kmax <= kmin:
        raise ValueError(f"kmax must be > kmin, got {kmax} <= {kmin}")
    if spacing.lower() == "log":
        return np.geomspace(kmin, kmax, nk)
    return np.linspace(kmin, kmax, nk)


@njit(cache=True)
def _log_gauss_window_scalar(x: float, x0: float, sigma_ln: float) -> float:
    if sigma_ln <= 0.0:
        return 1.0
    if x <= 0.0 or x0 <= 0.0:
        return 0.0
    z = np.log(x / x0) / sigma_ln
    return np.exp(-0.5 * z * z)


@njit(cache=True)
def _gamma_scalar(
    x: float,
    k: float,
    enabled: int,
    alpha: float,
    w_enabled: int,
    k_star: float,
    sigma_ln_k: float,
    xw_enabled: int,
    x0: float,
    sigma_ln_x: float,
) -> float:
    if enabled == 0 or alpha == 0.0:
        return 0.0

    xx = x if x > 1.0e-30 else 1.0e-30
    g = alpha / xx

    if w_enabled != 0:
        g *= _log_gauss_window_scalar(k, k_star, sigma_ln_k)
    if xw_enabled != 0:
        g *= _log_gauss_window_scalar(x, x0, sigma_ln_x)

    return g


@njit(cache=True)
def _deriv_numba(
    x: float,
    y: np.ndarray,
    k: float,
    enabled: int,
    alpha: float,
    w_enabled: int,
    k_star: float,
    sigma_ln_k: float,
    xw_enabled: int,
    x0: float,
    sigma_ln_x: float,
) -> np.ndarray:
    # y = [u_re, u_im, v_re, v_im], with v = du/dx
    u_re = y[0]
    u_im = y[1]
    v_re = y[2]
    v_im = y[3]

    g = _gamma_scalar(
        x, k, enabled, alpha, w_enabled, k_star, sigma_ln_k, xw_enabled, x0, sigma_ln_x
    )

    du_re = v_re
    du_im = v_im

    xx = x * x
    fac = -(1.0 - 2.0 / xx)

    dv_re = fac * u_re + g * v_re
    dv_im = fac * u_im + g * v_im

    out = np.empty(4, dtype=np.float64)
    out[0] = du_re
    out[1] = du_im
    out[2] = dv_re
    out[3] = dv_im
    return out


def solve_pk_memory_u(
    cfg_raw: Dict[str, Any],
    out_pk_path: Optional[str | Path] = None,
    diagnostics_path: Optional[str | Path] = None,
    return_arrays: bool = False,
) -> Dict[str, Any] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Friction-memory u-solver in x = -k eta.

    Baseline:
        u'' + (1 - 2/x^2) u = 0

    Modified:
        u'' + (1 - 2/x^2) u = + Gamma(x,k) u'
    """

    # -------------------------
    # Grid + primordial settings
    # -------------------------
    k_cfg = cfg_raw.get("k_grid", {})
    kmin = float(k_cfg.get("kmin", 1.0e-6))
    kmax = float(k_cfg.get("kmax", 1.0))
    nk = int(k_cfg.get("nk", 120))
    spacing = str(k_cfg.get("spacing", "log"))
    k_grid = _k_grid(kmin, kmax, nk, spacing)

    p_cfg = cfg_raw.get("primordial", {})
    As = float(p_cfg.get("As", 2.1e-9))
    pivot_k = float(p_cfg.get("pivot_k", 0.05))
    renorm = bool(p_cfg.get("renormalize_to_As", True))

    # -------------------------
    # Memory / friction parameters
    # -------------------------
    m_cfg = cfg_raw.get("memory", {})
    enabled = 1 if bool(m_cfg.get("enabled", False)) else 0
    alpha = float(m_cfg.get("alpha", 0.0))

    w_cfg = cfg_raw.get("window", {})
    w_enabled = 1 if bool(w_cfg.get("enabled", False)) else 0
    k_star = float(w_cfg.get("k_star", 0.002))
    sigma_ln_k = float(w_cfg.get("sigma_ln", 0.4))

    xw_cfg = cfg_raw.get("x_window", {})
    xw_enabled = 1 if bool(xw_cfg.get("enabled", False)) else 0
    x0 = float(xw_cfg.get("x0", 1.0))
    sigma_ln_x = float(xw_cfg.get("sigma_ln_x", 0.6))

    # -------------------------
    # Integration settings
    # -------------------------
    int_cfg = cfg_raw.get("integration", {})
    x_start = float(int_cfg.get("x_start", 80.0))
    x_end = float(int_cfg.get("x_end", 1.0e-3))
    method = str(int_cfg.get("method", "DOP853"))
    rtol = float(int_cfg.get("rtol", 1.0e-5))
    atol = float(int_cfg.get("atol", 1.0e-7))
    max_step = int_cfg.get("max_step", None)
    max_step = None if max_step in (None, "null") else float(max_step)
    max_step_val = np.inf if max_step is None else float(max_step)

    # -------------------------
    # Safety settings
    # -------------------------
    s_cfg = cfg_raw.get("safety", {})
    alpha_soft_cap = float(s_cfg.get("alpha_soft_cap", 1.2))
    max_rhs_calls = int(s_cfg.get("max_rhs_calls", 50000))
    reject_large_state = float(s_cfg.get("reject_large_state", 1.0e100))
    max_abs_gamma = float(s_cfg.get("max_abs_gamma", 1.0e4))
    slow_k_threshold_sec = float(s_cfg.get("slow_k_threshold_sec", 0.75))
    solver_wall_time_budget_sec = s_cfg.get("solver_wall_time_budget_sec", None)
    solver_wall_time_budget_sec = None if solver_wall_time_budget_sec in (None, "null") else float(solver_wall_time_budget_sec)
    per_k_wall_time_budget_sec = s_cfg.get("per_k_wall_time_budget_sec", None)
    per_k_wall_time_budget_sec = None if per_k_wall_time_budget_sec in (None, "null") else float(per_k_wall_time_budget_sec)
    max_slow_k_logs = int(s_cfg.get("max_slow_k_logs", 12))
    diag_log_path = s_cfg.get("solver_diag_log_path", None)

    # -------------------------
    # R conservation settings
    # -------------------------
    rcfg = cfg_raw.get("R_conservation", {})
    check_R_conservation = bool(rcfg.get("enabled", True))
    R_drift_threshold = float(rcfg.get("drift_threshold", 1.0e-3))
    R_monitor_nk_smallest = int(rcfg.get("monitor_nk_smallest", 5))
    R_tail_x_values = np.array(
        rcfg.get("tail_x_values", [1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3]),
        dtype=float,
    )

    if x_start <= x_end:
        raise ValueError(f"x_start must be > x_end, got {x_start} <= {x_end}")

    if enabled and abs(alpha) > alpha_soft_cap:
        raise MemorySolverFailure(
            f"Rejected point before integration: |alpha|={abs(alpha):.3e} > {alpha_soft_cap:.3e}"
        )

    pk_raw = np.zeros_like(k_grid, dtype=float)
    freeze_ratio = np.zeros_like(k_grid, dtype=float)
    Rabs = np.zeros_like(k_grid, dtype=float)
    nfev_used = np.zeros_like(k_grid, dtype=np.int64)
    k_wall_sec = np.zeros_like(k_grid, dtype=float)

    smallk_R_drift = []
    slow_k_log_count = 0
    t_total0 = perf_counter()

    for i, k in enumerate(k_grid):
        if solver_wall_time_budget_sec is not None:
            total_elapsed_pre = perf_counter() - t_total0
            if total_elapsed_pre > solver_wall_time_budget_sec:
                raise MemorySolverFailure(
                    f"Solver wall-clock budget exceeded before k-loop advance: "
                    f"elapsed={total_elapsed_pre:.3f}s > budget={solver_wall_time_budget_sec:.3f}s"
                )

        rhs_calls = 0
        t_k0 = perf_counter()

        u0 = np.exp(1j * x_start) / np.sqrt(2.0)
        v0 = 1j * np.exp(1j * x_start) / np.sqrt(2.0)
        y0 = np.array([u0.real, u0.imag, v0.real, v0.imag], dtype=np.float64)

        def fun(x, y, _k=k):
            nonlocal rhs_calls
            rhs_calls += 1

            now = perf_counter()

            if solver_wall_time_budget_sec is not None:
                total_elapsed = now - t_total0
                if total_elapsed > solver_wall_time_budget_sec:
                    raise MemorySolverFailure(
                        f"Solver wall-clock budget exceeded at k={_k:.3e}: "
                        f"elapsed={total_elapsed:.3f}s > budget={solver_wall_time_budget_sec:.3f}s"
                    )

            if per_k_wall_time_budget_sec is not None:
                k_elapsed = now - t_k0
                if k_elapsed > per_k_wall_time_budget_sec:
                    raise MemorySolverFailure(
                        f"Per-k wall-clock budget exceeded at k={_k:.3e}: "
                        f"elapsed={k_elapsed:.3f}s > budget={per_k_wall_time_budget_sec:.3f}s"
                    )

            if rhs_calls > max_rhs_calls:
                raise MemorySolverFailure(
                    f"Exceeded RHS budget at k={_k:.3e}, calls={rhs_calls}, method={method}"
                )

            if not np.all(np.isfinite(y)):
                raise MemorySolverFailure(f"Non-finite state at k={_k:.3e}")

            if np.any(np.abs(y) > reject_large_state):
                raise MemorySolverFailure(f"State blow-up at k={_k:.3e}")

            g_here = _gamma_scalar(
                float(x),
                float(_k),
                enabled,
                alpha,
                w_enabled,
                k_star,
                sigma_ln_k,
                xw_enabled,
                x0,
                sigma_ln_x,
            )
            if (not np.isfinite(g_here)) or abs(g_here) > max_abs_gamma:
                raise MemorySolverFailure(
                    f"Gamma instability at k={_k:.3e}, x={x:.3e}, gamma={g_here}"
                )

            dy = _deriv_numba(
                float(x),
                y,
                float(_k),
                enabled,
                alpha,
                w_enabled,
                k_star,
                sigma_ln_k,
                xw_enabled,
                x0,
                sigma_ln_x,
            )

            if not np.all(np.isfinite(dy)):
                raise MemorySolverFailure(f"Non-finite derivative at k={_k:.3e}")

            return dy

        try:
            sol = solve_ivp(
                fun=fun,
                t_span=(x_start, x_end),
                y0=y0,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step_val,
                dense_output=True,
            )
        except MemorySolverFailure:
            raise
        except Exception as e:
            raise MemorySolverFailure(f"solve_ivp crashed for k={k:.3e}: {e}") from e

        dt_k = perf_counter() - t_k0
        k_wall_sec[i] = dt_k

        if per_k_wall_time_budget_sec is not None and dt_k > per_k_wall_time_budget_sec:
            raise MemorySolverFailure(
                f"Per-k wall-clock budget exceeded after solve at k={k:.3e}: "
                f"elapsed={dt_k:.3f}s > budget={per_k_wall_time_budget_sec:.3f}s"
            )

        if dt_k > slow_k_threshold_sec and slow_k_log_count < max_slow_k_logs:
            y_end_tmp = sol.y[:, -1]
            u_end_tmp = y_end_tmp[0] + 1j * y_end_tmp[1]
            v_end_tmp = y_end_tmp[2] + 1j * y_end_tmp[3]
            R_end_tmp = x_end * u_end_tmp
            Rp_end_tmp = x_end * v_end_tmp + u_end_tmp
            denom_tmp = np.abs(R_end_tmp)
            fr_tmp = float(np.abs(Rp_end_tmp) / denom_tmp) if denom_tmp > 0 else np.nan
            _append_jsonl(
                diag_log_path,
                {
                    "event": "k_slow",
                    "alpha": alpha,
                    "k_star": k_star,
                    "sigma_ln": sigma_ln_k,
                    "method": method,
                    "k": float(k),
                    "rhs_calls": int(rhs_calls),
                    "dt_sec": float(dt_k),
                    "freeze_ratio": fr_tmp,
                    "Rabs": float(np.abs(R_end_tmp)),
                },
            )
            slow_k_log_count += 1

        if (not sol.success) or sol.y.shape[1] == 0:
            raise MemorySolverFailure(f"u-memory ODE failed for k={k:.3e}: {sol.message}")

        y_end = sol.y[:, -1]
        if not np.all(np.isfinite(y_end)):
            raise MemorySolverFailure(f"Non-finite final state for k={k:.3e}")

        u_end = y_end[0] + 1j * y_end[1]
        v_end = y_end[2] + 1j * y_end[3]

        R_end = x_end * u_end
        Rp_end = x_end * v_end + u_end

        Rabs[i] = float(np.abs(R_end))
        denom = np.abs(R_end)
        freeze_ratio[i] = float(np.abs(Rp_end) / denom) if denom > 0 else np.nan

        pk_val = float((k**3 / (2.0 * np.pi**2)) * (np.abs(R_end) ** 2))
        if (not np.isfinite(pk_val)) or pk_val <= 0.0:
            raise MemorySolverFailure(f"Non-finite or non-positive pk_raw at k={k:.3e}")

        pk_raw[i] = pk_val
        nfev_used[i] = rhs_calls

        if check_R_conservation and i < min(R_monitor_nk_smallest, len(k_grid)):
            if sol.sol is None:
                raise MemorySolverFailure(f"Dense output missing at k={k:.3e} for R conservation check")

            tail_x = R_tail_x_values[(R_tail_x_values <= x_start) & (R_tail_x_values >= x_end)]
            if tail_x.size >= 2:
                y_tail = sol.sol(tail_x)
                u_tail = y_tail[0] + 1j * y_tail[1]
                R_tail = tail_x * u_tail
                R_ref = np.abs(R_tail[-1]) + 1.0e-300
                drift = float(np.max(np.abs(R_tail - R_tail[-1]) / R_ref))
                smallk_R_drift.append(drift)

    if check_R_conservation and len(smallk_R_drift) > 0:
        drift_max = float(np.max(smallk_R_drift))
        if drift_max > R_drift_threshold:
            raise MemorySolverFailure(
                f"Super-horizon R drift too large: max drift={drift_max:.3e} > {R_drift_threshold:.3e}"
            )
    else:
        drift_max = float("nan")

    renorm_factor = 1.0
    if renorm:
        Pp = float(np.interp(pivot_k, k_grid, pk_raw))
        if (not np.isfinite(Pp)) or Pp <= 0:
            raise MemorySolverFailure(f"Bad pivot power P(pivot)={Pp} for pivot_k={pivot_k}")
        renorm_factor = As / Pp

    pk = renorm_factor * pk_raw
    if (not np.isfinite(pk).all()) or (pk <= 0).any():
        raise MemorySolverFailure("Renormalized spectrum is non-finite or non-positive")

    def _p95(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.percentile(x, 95.0)) if x.size else float("nan")

    total_wall_sec = float(perf_counter() - t_total0)

    diag: Dict[str, Any] = {
        "As": As,
        "pivot_k": pivot_k,
        "renormalize_to_As": renorm,
        "renorm_factor": float(renorm_factor),
        "memory_enabled": bool(enabled),
        "alpha": float(alpha),
        "window_enabled": bool(w_enabled),
        "k_star": float(k_star),
        "sigma_ln": float(sigma_ln_k),
        "x_window_enabled": bool(xw_enabled),
        "x0": float(x0),
        "sigma_ln_x": float(sigma_ln_x),
        "x_start": float(x_start),
        "x_end": float(x_end),
        "method": method,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_step": (None if max_step is None else float(max_step)),
        "freeze_ratio_max": float(np.nanmax(freeze_ratio)),
        "freeze_ratio_p95": _p95(freeze_ratio),
        "deltaR_median": float(np.nanmedian(Rabs)),
        "deltaR_p95": _p95(Rabs),
        "nfev_max": int(np.max(nfev_used)),
        "nfev_p95": _p95(nfev_used.astype(float)),
        "k_time_max_sec": float(np.max(k_wall_sec)),
        "k_time_p95_sec": _p95(k_wall_sec),
        "total_wall_sec": total_wall_sec,
        "R_tail_drift_max": float(drift_max),
        "solver_wall_time_budget_sec": (
            None if solver_wall_time_budget_sec is None else float(solver_wall_time_budget_sec)
        ),
        "per_k_wall_time_budget_sec": (
            None if per_k_wall_time_budget_sec is None else float(per_k_wall_time_budget_sec)
        ),
    }

    _append_jsonl(
        diag_log_path,
        {
            "event": "spectrum_done",
            "alpha": alpha,
            "k_star": k_star,
            "sigma_ln": sigma_ln_k,
            "method": method,
            "freeze_ratio_p95": diag["freeze_ratio_p95"],
            "nfev_max": diag["nfev_max"],
            "k_time_max_sec": diag["k_time_max_sec"],
            "R_tail_drift_max": diag["R_tail_drift_max"],
            "total_wall_sec": diag["total_wall_sec"],
        },
    )

    if out_pk_path is not None:
        out_pk_path = Path(out_pk_path)
        out_pk_path.parent.mkdir(parents=True, exist_ok=True)
        header = "# k_Mpc^-1    P_R(k)"
        np.savetxt(out_pk_path, np.c_[k_grid, pk], header=header)

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            diagnostics_path,
            np.c_[k_grid, freeze_ratio, nfev_used, k_wall_sec],
            header="# k_Mpc^-1 freeze_ratio nfev k_wall_sec",
        )

        try:
            run_tag = diagnostics_path.name.replace("freeze_diag_", "").replace(".txt", "")
            dpath = diagnostics_path.parent / f"deltaR_diag_{run_tag}.txt"
        except Exception:
            dpath = diagnostics_path.parent / "deltaR_diag.txt"
        np.savetxt(dpath, np.c_[k_grid, Rabs], header="# k_Mpc^-1 |R_end|")

    if return_arrays:
        return k_grid, pk, diag
    return diag

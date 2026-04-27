from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
from cobaya.theory import Theory

from .memory_u_solver import MemorySolverFailure, solve_pk_memory_u


def _append_jsonl(path: Optional[str], row: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


class bml_theory(Theory):
    """
    Provides primordial_scalar_pk for Cobaya CAMB with external_primordial_pk=True.

    Strategy:
      1) Solve ODE twice with As=1 and no internal renormalization:
         - memory enabled  -> P_raw_mem(k)
         - memory disabled -> P_raw_base(k), cached per MPI rank
      2) shape_ratio = P_raw_mem / P_raw_base
      3) Dress with physical amplitude and tilt:
         P_R(k) = As * (k/pivot_k)^(ns-1) * shape_ratio
    """

    params = {"As": None, "ns": None, "alpha": None, "k_star": None, "sigma_ln": None}

    kmin: float = 1.0e-6
    kmax: float = 1.0
    nk: int = 120
    spacing: str = "log"
    pivot_k: float = 0.05

    x_start: float = 80.0
    x_end: float = 1.0e-3
    method: str = "DOP853"
    rtol: float = 1.0e-5
    atol: float = 1.0e-7
    max_step: Optional[float] = None

    window_enabled: bool = True
    x_window_enabled: bool = True
    x0: float = 1.0
    sigma_ln_x: float = 0.6

    pk_kind: str = "auto"

    alpha_zero_tol: float = 1.0e-14
    alpha_soft_cap: float = 1.0
    min_shape_ratio: float = 1.0e-12
    max_shape_ratio: float = 1.0e12
    max_shape_dynamic_range: float = 1.0e8
    max_log_slope_abs: float = 25.0

    theory_diag_log_path: Optional[str] = None
    max_calculate_wall_time_sec: Optional[float] = 90.0
    solver_wall_time_budget_sec: Optional[float] = 45.0
    per_k_wall_time_budget_sec: Optional[float] = 2.0

    cache_memory_spectra: bool = True
    memory_cache_maxsize: int = 64
    cache_round_ndigits: int = 12

    check_R_conservation: bool = True
    R_drift_threshold: float = 1.0e-3
    R_monitor_nk_smallest: int = 5
    R_tail_x_values: list[float] = [1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3]

    def initialize(self):
        self._last_pk: Optional[Dict[str, Any]] = None
        self._base_cache: Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = None
        self._memory_cache: Dict[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}

    def get_requirements(self):
        return {}

    def get_can_provide(self):
        return ["primordial_scalar_pk"]

    def _rounded_key(self, alpha: float, k_star: float, sigma_ln: float) -> Tuple[float, float, float]:
        nd = int(self.cache_round_ndigits)
        return (round(float(alpha), nd), round(float(k_star), nd), round(float(sigma_ln), nd))

    def _evict_memory_cache_if_needed(self) -> None:
        maxsize = int(self.memory_cache_maxsize)
        if maxsize <= 0:
            self._memory_cache.clear()
            return
        while len(self._memory_cache) > maxsize:
            first_key = next(iter(self._memory_cache))
            del self._memory_cache[first_key]

    def _integration_cfg(self) -> Dict[str, Any]:
        cfg = {
            "x_start": float(self.x_start),
            "x_end": float(self.x_end),
            "method": str(self.method),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
        }
        if self.max_step is not None:
            cfg["max_step"] = float(self.max_step)
        return cfg

    def _safety_cfg(self) -> Dict[str, Any]:
        return {
            "alpha_soft_cap": float(self.alpha_soft_cap),
            "max_rhs_calls": 50000,
            "reject_large_state": 1.0e100,
            "max_abs_gamma": 1.0e4,
            "slow_k_threshold_sec": 0.75,
            "solver_diag_log_path": self.theory_diag_log_path,
            "solver_wall_time_budget_sec": self.solver_wall_time_budget_sec,
            "per_k_wall_time_budget_sec": self.per_k_wall_time_budget_sec,
            "max_slow_k_logs": 12,
        }

    def _R_conservation_cfg(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.check_R_conservation),
            "drift_threshold": float(self.R_drift_threshold),
            "monitor_nk_smallest": int(self.R_monitor_nk_smallest),
            "tail_x_values": list(self.R_tail_x_values),
        }

    def _common_cfg(self, k_star: float, sigma_ln: float) -> Dict[str, Any]:
        return {
            "k_grid": {
                "kmin": float(self.kmin),
                "kmax": float(self.kmax),
                "nk": int(self.nk),
                "spacing": str(self.spacing),
            },
            "primordial": {
                "As": 1.0,
                "pivot_k": float(self.pivot_k),
                "renormalize_to_As": False,
            },
            "window": {
                "enabled": bool(self.window_enabled),
                "k_star": float(k_star),
                "sigma_ln": float(sigma_ln),
            },
            "x_window": {
                "enabled": bool(self.x_window_enabled),
                "x0": float(self.x0),
                "sigma_ln_x": float(self.sigma_ln_x),
            },
            "integration": self._integration_cfg(),
            "safety": self._safety_cfg(),
            "R_conservation": self._R_conservation_cfg(),
            "pk_kind": str(self.pk_kind),
        }

    def _validate_positive_array(self, name: str, arr: np.ndarray):
        if arr.ndim != 1 or arr.size < 2:
            raise RuntimeError(f"{name} must be 1D with length >= 2")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{name} contains non-finite values")
        if (arr <= 0).any():
            raise RuntimeError(f"{name} contains non-positive values")

    def _validate_log_slope(self, k: np.ndarray, pk: np.ndarray) -> bool:
        logk = np.log(k)
        logpk = np.log(pk)
        slope = np.gradient(logpk, logk)
        return bool(np.all(np.isfinite(slope)) and np.max(np.abs(slope)) <= float(self.max_log_slope_abs))

    def _enforce_calculate_budget(self, t0: float, stage: str) -> None:
        if self.max_calculate_wall_time_sec is None:
            return
        dt = perf_counter() - t0
        if dt > float(self.max_calculate_wall_time_sec):
            raise MemorySolverFailure(
                f"calculate wall-clock budget exceeded after {stage}: "
                f"elapsed={dt:.3f}s > budget={float(self.max_calculate_wall_time_sec):.3f}s"
            )

    def _get_base(self, tcalc0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if self._base_cache is not None:
            _append_jsonl(
                self.theory_diag_log_path,
                {
                    "event": "base_cache_hit",
                    "method": self.method,
                    "nk": int(self.nk),
                },
            )
            if tcalc0 is not None:
                self._enforce_calculate_budget(tcalc0, "base_cache_hit")
            return self._base_cache

        t0 = perf_counter()
        _append_jsonl(
            self.theory_diag_log_path,
            {
                "event": "base_start",
                "method": self.method,
                "nk": int(self.nk),
            },
        )

        cfg_base = {
            "k_grid": {
                "kmin": float(self.kmin),
                "kmax": float(self.kmax),
                "nk": int(self.nk),
                "spacing": str(self.spacing),
            },
            "primordial": {
                "As": 1.0,
                "pivot_k": float(self.pivot_k),
                "renormalize_to_As": False,
            },
            "memory": {"enabled": False, "alpha": 0.0},
            "window": {
                "enabled": bool(self.window_enabled),
                "k_star": 0.002,
                "sigma_ln": 0.4,
            },
            "x_window": {
                "enabled": bool(self.x_window_enabled),
                "x0": float(self.x0),
                "sigma_ln_x": float(self.sigma_ln_x),
            },
            "integration": self._integration_cfg(),
            "safety": self._safety_cfg(),
            "R_conservation": self._R_conservation_cfg(),
            "pk_kind": str(self.pk_kind),
        }

        ks, P_base, diag_base = solve_pk_memory_u(cfg_base, return_arrays=True)
        self._validate_positive_array("baseline spectrum", P_base)
        self._base_cache = (ks, P_base, diag_base)

        dt = perf_counter() - t0
        _append_jsonl(
            self.theory_diag_log_path,
            {
                "event": "base_done",
                "dt_sec": dt,
                "method": self.method,
                "nk": int(self.nk),
                "nfev_max": diag_base.get("nfev_max", np.nan),
                "k_time_max_sec": diag_base.get("k_time_max_sec", np.nan),
                "R_tail_drift_max": diag_base.get("R_tail_drift_max", np.nan),
            },
        )

        if tcalc0 is not None:
            self._enforce_calculate_budget(tcalc0, "base_done")

        return self._base_cache

    def _get_memory_spectrum(
        self,
        alpha: float,
        k_star: float,
        sigma_ln: float,
        tcalc0: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], bool]:
        cache_key = self._rounded_key(alpha, k_star, sigma_ln)

        if self.cache_memory_spectra and cache_key in self._memory_cache:
            ks, P_mem, diag_mem = self._memory_cache[cache_key]
            _append_jsonl(
                self.theory_diag_log_path,
                {
                    "event": "memory_cache_hit",
                    "alpha": alpha,
                    "k_star": k_star,
                    "sigma_ln": sigma_ln,
                },
            )
            self._enforce_calculate_budget(tcalc0, "memory_cache_hit")
            return ks, P_mem, diag_mem, True

        cfg_common = self._common_cfg(k_star=k_star, sigma_ln=sigma_ln)
        cfg_mem = dict(cfg_common)
        cfg_mem["memory"] = {"enabled": True, "alpha": float(alpha)}

        _append_jsonl(
            self.theory_diag_log_path,
            {
                "event": "memory_start",
                "alpha": alpha,
                "k_star": k_star,
                "sigma_ln": sigma_ln,
            },
        )

        t0 = perf_counter()
        ks, P_mem, diag_mem = solve_pk_memory_u(cfg_mem, return_arrays=True)
        dt = perf_counter() - t0

        self._validate_positive_array("memory spectrum", P_mem)

        _append_jsonl(
            self.theory_diag_log_path,
            {
                "event": "memory_done",
                "alpha": alpha,
                "k_star": k_star,
                "sigma_ln": sigma_ln,
                "dt_sec": dt,
                "nfev_max": diag_mem.get("nfev_max", np.nan),
                "k_time_max_sec": diag_mem.get("k_time_max_sec", np.nan),
                "R_tail_drift_max": diag_mem.get("R_tail_drift_max", np.nan),
            },
        )

        if self.cache_memory_spectra:
            self._memory_cache[cache_key] = (ks, P_mem, diag_mem)
            self._evict_memory_cache_if_needed()

        self._enforce_calculate_budget(tcalc0, "memory_done")
        return ks, P_mem, diag_mem, False

    def calculate(self, state: Dict[str, Any], want_derived: bool = False, **p):
        t0 = perf_counter()

        As = float(p.get("As", 2.1e-9))
        ns = float(p.get("ns", 0.965))
        alpha = float(p.get("alpha", 0.0))
        k_star = float(p.get("k_star", 0.002))
        sigma_ln = float(p.get("sigma_ln", 0.4))

        _append_jsonl(
            self.theory_diag_log_path,
            {
                "event": "calc_start",
                "As": As,
                "ns": ns,
                "alpha": alpha,
                "k_star": k_star,
                "sigma_ln": sigma_ln,
            },
        )

        try:
            if abs(alpha) > float(self.alpha_soft_cap):
                _append_jsonl(
                    self.theory_diag_log_path,
                    {
                        "event": "reject",
                        "reason": "alpha_soft_cap",
                        "alpha": alpha,
                        "alpha_soft_cap": self.alpha_soft_cap,
                    },
                )
                return False

            ks_base, P_base, diag_base = self._get_base(tcalc0=t0)

            if abs(alpha) <= float(self.alpha_zero_tol):
                shape_ratio = np.ones_like(P_base)
                diag_mem = dict(diag_base)
                diag_mem["shortcut_alpha_zero"] = True
                memory_cache_hit = False
            else:
                try:
                    ks, P_mem, diag_mem, memory_cache_hit = self._get_memory_spectrum(
                        alpha=alpha,
                        k_star=k_star,
                        sigma_ln=sigma_ln,
                        tcalc0=t0,
                    )
                except MemorySolverFailure as e:
                    _append_jsonl(
                        self.theory_diag_log_path,
                        {
                            "event": "reject",
                            "reason": "memory_solver_failure",
                            "alpha": alpha,
                            "k_star": k_star,
                            "sigma_ln": sigma_ln,
                            "message": str(e),
                        },
                    )
                    return False
                except Exception as e:
                    _append_jsonl(
                        self.theory_diag_log_path,
                        {
                            "event": "reject",
                            "reason": "unexpected_solver_exception",
                            "alpha": alpha,
                            "k_star": k_star,
                            "sigma_ln": sigma_ln,
                            "message": repr(e),
                        },
                    )
                    return False

                if ks.shape != ks_base.shape or (not np.allclose(ks, ks_base, rtol=0.0, atol=0.0)):
                    raise RuntimeError("Baseline and memory k-grids do not match")

                shape_ratio = P_mem / np.maximum(P_base, 1.0e-60)

                if not np.isfinite(shape_ratio).all():
                    _append_jsonl(self.theory_diag_log_path, {"event": "reject", "reason": "nonfinite_shape_ratio"})
                    return False
                if (shape_ratio <= 0).any():
                    _append_jsonl(self.theory_diag_log_path, {"event": "reject", "reason": "nonpositive_shape_ratio"})
                    return False
                if (shape_ratio < float(self.min_shape_ratio)).any():
                    _append_jsonl(
                        self.theory_diag_log_path,
                        {
                            "event": "reject",
                            "reason": "shape_ratio_too_small",
                            "shape_ratio_min": float(np.min(shape_ratio)),
                        },
                    )
                    return False
                if (shape_ratio > float(self.max_shape_ratio)).any():
                    _append_jsonl(
                        self.theory_diag_log_path,
                        {
                            "event": "reject",
                            "reason": "shape_ratio_too_large",
                            "shape_ratio_max": float(np.max(shape_ratio)),
                        },
                    )
                    return False

                sr_min = float(np.min(shape_ratio))
                sr_max = float(np.max(shape_ratio))
                if sr_min <= 0.0 or (sr_max / sr_min) > float(self.max_shape_dynamic_range):
                    _append_jsonl(
                        self.theory_diag_log_path,
                        {
                            "event": "reject",
                            "reason": "shape_dynamic_range",
                            "shape_ratio_min": sr_min,
                            "shape_ratio_max": sr_max,
                        },
                    )
                    return False

            self._enforce_calculate_budget(t0, "shape_construction")

            tilt = (ks_base / float(self.pivot_k)) ** (ns - 1.0)
            P_physical = As * tilt * shape_ratio

            if (not np.isfinite(P_physical).all()) or (P_physical <= 0).any():
                _append_jsonl(self.theory_diag_log_path, {"event": "reject", "reason": "bad_physical_pk"})
                return False

            if not self._validate_log_slope(ks_base, P_physical):
                _append_jsonl(self.theory_diag_log_path, {"event": "reject", "reason": "log_slope_too_large"})
                return False

            self._last_pk = {
                "k": ks_base,
                "Pk": P_physical,
                "log_regular": (str(self.spacing).lower() == "log"),
                "kmin": float(ks_base[0]),
                "kmax": float(ks_base[-1]),
            }
            state["primordial_scalar_pk"] = self._last_pk

            if want_derived:
                state["derived"] = {
                    "shape_ratio_p50": float(np.median(shape_ratio)),
                    "shape_ratio_p05": float(np.quantile(shape_ratio, 0.05)),
                    "shape_ratio_p95": float(np.quantile(shape_ratio, 0.95)),
                    "shape_ratio_min": float(np.min(shape_ratio)),
                    "shape_ratio_max": float(np.max(shape_ratio)),
                    "diag_mem_freeze_p95": float(diag_mem.get("freeze_ratio_p95", np.nan)),
                    "diag_base_freeze_p95": float(diag_base.get("freeze_ratio_p95", np.nan)),
                    "diag_mem_R_tail_drift_max": float(diag_mem.get("R_tail_drift_max", np.nan)),
                    "diag_base_R_tail_drift_max": float(diag_base.get("R_tail_drift_max", np.nan)),
                }

            dt_total = perf_counter() - t0
            _append_jsonl(
                self.theory_diag_log_path,
                {
                    "event": "calc_done",
                    "alpha": alpha,
                    "k_star": k_star,
                    "sigma_ln": sigma_ln,
                    "shape_ratio_min": float(np.min(shape_ratio)),
                    "shape_ratio_max": float(np.max(shape_ratio)),
                    "dt_sec": dt_total,
                    "memory_cache_hit": bool(memory_cache_hit),
                },
            )

            return True

        except MemorySolverFailure as e:
            _append_jsonl(
                self.theory_diag_log_path,
                {
                    "event": "reject",
                    "reason": "calculate_budget_or_solver_failure",
                    "alpha": alpha,
                    "k_star": k_star,
                    "sigma_ln": sigma_ln,
                    "message": str(e),
                },
            )
            return False

    def get_primordial_scalar_pk(self):
        return self._last_pk

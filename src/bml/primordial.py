from __future__ import annotations
import numpy as np

def make_k_grid(kmin: float, kmax: float, nk: int, spacing: str = "log") -> np.ndarray:
    if spacing.lower() == "log":
        return np.logspace(np.log10(kmin), np.log10(kmax), nk)
    if spacing.lower() == "linear":
        return np.linspace(kmin, kmax, nk)
    raise ValueError(f"Unknown spacing: {spacing}")

def power_law_pk(k: np.ndarray, As: float, ns: float, pivot_k: float) -> np.ndarray:
    return As * (k / pivot_k) ** (ns - 1.0)

def save_pk_table(path: str, k: np.ndarray, pk: np.ndarray) -> None:
    header = "k_Mpc^-1    P_R(k)"
    data = np.column_stack([k, pk])
    np.savetxt(path, data, header=header)

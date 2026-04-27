from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_pk(k: np.ndarray, pk: np.ndarray, outpath: Path, title: str) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.loglog(k, pk)
    plt.xlabel("k [Mpc^-1]")
    plt.ylabel("P_R(k)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

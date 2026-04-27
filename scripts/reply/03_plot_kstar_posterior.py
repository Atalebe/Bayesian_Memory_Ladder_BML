import numpy as np
import matplotlib.pyplot as plt
from _paths import RUN, FIG

arr = np.loadtxt(RUN["0116"]["equal_weights"])
kstar = arr[:, 9]

p16, p50, p84 = np.percentile(kstar, [16, 50, 84])

fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.hist(kstar, bins=20)
ax.axvline(p16, linestyle="--", label="16th")
ax.axvline(p50, linestyle="-", label="median")
ax.axvline(p84, linestyle="--", label="84th")
ax.set_xlabel(r"$k_\star\ [{\rm Mpc}^{-1}]$")
ax.set_ylabel("count")
ax.set_title(r"Restricted-memory posterior for $k_\star$")
ax.legend()
out = FIG / "fig02_kstar_posterior.png"
fig.tight_layout()
fig.savefig(out, dpi=200)
print(f"[ok] wrote {out}")
print(f"kstar median={p50:.6e}, p16={p16:.6e}, p84={p84:.6e}")

import numpy as np
import matplotlib.pyplot as plt
from _paths import RUN, FIG

arr15 = np.loadtxt(RUN["0115"]["equal_weights"])
arr16 = np.loadtxt(RUN["0116"]["equal_weights"])

h015 = arr15[:, 4]
h016 = arr16[:, 4]

p15 = np.percentile(h015, [16, 50, 84])
p16 = np.percentile(h016, [16, 50, 84])

fig, ax = plt.subplots(figsize=(6.6, 4.6))
x = [0, 1]
y = [p15[1], p16[1]]
yerr = [[p15[1] - p15[0], p16[1] - p16[0]], [p15[2] - p15[1], p16[2] - p16[1]]]
ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(["0115 baseline", "0116 restricted"])
ax.set_ylabel(r"$H_0\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$")
ax.set_title(r"$H_0$ comparison")
out = FIG / "figA_h0_comparison.png"
fig.tight_layout()
fig.savefig(out, dpi=200)

print("0115", p15[1])
print("0116", p16[1])
print(f"[ok] wrote {out}")

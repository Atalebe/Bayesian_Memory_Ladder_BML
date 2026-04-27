import re
import matplotlib.pyplot as plt
from _paths import RUN, FIG

PAT = re.compile(r"log\(Z\)\s*=\s*([-\d.E+]+)\s*\+/-\s*([-\d.E+]+)")

def read_logz(path):
    txt = path.read_text()
    m = PAT.search(txt)
    if not m:
        raise RuntimeError(f"Could not parse logZ from {path}")
    return float(m.group(1)), float(m.group(2))

z15, s15 = read_logz(RUN["0115"]["stats"])
z16, s16 = read_logz(RUN["0116"]["stats"])
delta = z16 - z15

labels = ["0115 baseline", "0116 restricted"]
vals = [z15, z16]
errs = [s15, s16]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = [0, 1]
ax.bar(x, vals, yerr=errs, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("log(Z)")
ax.set_title("Active Bayesian Evidence Comparison")
ax.text(0.5, max(vals) + 1.0, f"ΔlogZ(0116-0115) = {delta:.2f}", ha="center")
out = FIG / "fig01_evidence_comparison.png"
fig.tight_layout()
fig.savefig(out, dpi=200)
print(f"[ok] wrote {out}")

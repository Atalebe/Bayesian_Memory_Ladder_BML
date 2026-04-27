import gzip
import json
from collections import Counter
import matplotlib.pyplot as plt
from _paths import RUN, FIG, TAB

accepted_alpha = []
accepted_kstar = []
accepted_shape = []
reject_alpha = []
reject_kstar = []
reasons = Counter()

with gzip.open(RUN["0116"]["diag_gz"], "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        ev = row.get("event")
        if ev == "calc_done":
            accepted_alpha.append(row.get("alpha"))
            accepted_kstar.append(row.get("k_star"))
            accepted_shape.append(row.get("shape_ratio_min"))
        elif ev == "reject":
            reject_alpha.append(row.get("alpha"))
            reject_kstar.append(row.get("k_star"))
            reasons[row.get("reason", "UNKNOWN")] += 1

fig, ax = plt.subplots(figsize=(7.2, 5.0))
sc = ax.scatter(accepted_kstar, accepted_alpha, c=accepted_shape, s=10, alpha=0.7)
if reject_alpha:
    ax.scatter(reject_kstar, reject_alpha, marker="x", s=18, label="rejects")
ax.set_xlabel(r"$k_\star\ [{\rm Mpc}^{-1}]$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Restricted-memory numerical audit")
if reject_alpha:
    ax.legend()
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("shape_ratio_min")
out = FIG / "fig03_numerical_audit.png"
fig.tight_layout()
fig.savefig(out, dpi=200)

summary = TAB / "numerical_audit_summary.txt"
with summary.open("w") as f:
    total = len(accepted_alpha) + len(reject_alpha)
    frac = len(reject_alpha) / total if total else float("nan")
    f.write(f"accepted={len(accepted_alpha)}\n")
    f.write(f"rejected={len(reject_alpha)}\n")
    f.write(f"reject_fraction={frac}\n")
    for k, v in reasons.items():
        f.write(f"{k}={v}\n")

print(f"[ok] wrote {out}")
print(f"[ok] wrote {summary}")

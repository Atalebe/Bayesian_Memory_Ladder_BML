import csv
import re
from _paths import RUN, TAB

PAT = re.compile(r"log\(Z\)\s*=\s*([-\d.E+]+)\s*\+/-\s*([-\d.E+]+)")

def read_logz(path):
    txt = path.read_text()
    m = PAT.search(txt)
    if not m:
        raise RuntimeError(f"Could not parse logZ from {path}")
    return float(m.group(1)), float(m.group(2))

rows = []
base_logz, base_sigma = read_logz(RUN["0115"]["stats"])

for run in ["0115", "0116", "0118"]:
    logz, sigma = read_logz(RUN[run]["stats"])
    delta = logz - base_logz
    if run == "0115":
        note = "baseline reference"
    elif run == "0116":
        note = "very strong active lead" if delta > 5 else "active lead"
    else:
        note = "controlled broad-memory comparison"
    rows.append((run, logz, sigma, delta, note))

csv_path = TAB / "table01_evidence_ledger.csv"
tex_path = TAB / "table01_evidence_ledger.tex"

with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "logZ", "sigma", "delta_vs_0115", "note"])
    for r in rows:
        w.writerow(r)

with tex_path.open("w") as f:
    f.write("\\begin{tabular}{lrrrrl}\n")
    f.write("\\toprule\n")
    f.write("Run & $\\log Z$ & $\\sigma_Z$ & $\\Delta\\log Z$ & & Note \\\\\n")
    f.write("\\midrule\n")
    for run, logz, sigma, delta, note in rows:
        f.write(f"{run} & {logz:.5f} & {sigma:.5f} & {delta:.5f} & & {note} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print(f"[ok] wrote {csv_path}")
print(f"[ok] wrote {tex_path}")
for r in rows:
    print(*r)

import gzip, json
from collections import Counter
from pathlib import Path

path = Path("artifacts/0116/run_0116_bml_theory_diag_snapshot_2026-04-27.jsonl.gz")

accepted = 0
rejected = 0
reasons = Counter()

with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        ev = row.get("event")
        if ev == "calc_done":
            accepted += 1
        elif ev == "reject":
            rejected += 1
            reasons[row.get("reason", "UNKNOWN")] += 1

print("accepted =", accepted)
print("rejected =", rejected)
print("reject_fraction =", rejected / (accepted + rejected))
print("reject_reasons =", dict(reasons))

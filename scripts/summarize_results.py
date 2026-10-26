from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# Ditambahkan "map50_s" dan "map50_m" ke dalam daftar METRICS
METRICS = ["map50", "map50_95", "map50_s", "map50_m", "precision", "recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv")
    a = ap.parse_args()

    groups = defaultdict(list)
    for f in sorted(Path(a.runs).glob("*/seed*/result.json")):
        r = json.loads(f.read_text())
        groups[r["name"]].append(r)
    if not groups:
        raise SystemExit(f"tidak ada result.json di {a.runs}/*/seed*/")

    def ms(rs, k):
        v = [r.get(k, 0.0) * 100 for r in rs]
        return (st.mean(v), st.pstdev(v) if len(v) > 1 else float("nan"))

    rows = []
    for name, rs in sorted(groups.items(), key=lambda kv: (kv[1][0].get("stage") or 0, kv[0])):
        r0 = rs[0]
        row = {"name": name, "stage": r0.get("stage"), "arch": r0["arch"], "imgsz": r0["imgsz"], "attn": r0["attn"],
               "loss": f"{r0['loss']['iou']}" + (f"+nwd{r0['loss']['nwd_ratio']}" if r0["loss"]["nwd_ratio"] else ""),
               "seeds": len(rs), "params_M": r0["params_M"], "gflops": r0["gflops"]}  # fmt: skip
        for k in METRICS:
            row[k], row[k + "_std"] = ms(rs, k)
        rows.append(row)

    # Menyesuaikan tampilan header terminal agar mAP50_s dan mAP50_m muat
    print(f"{'nama':18s} {'arch':8s} {'imgsz':>5s} {'attn':>5s} {'loss':>12s} | {'mAP50':>12s} {'mAP_s':>12s} {'mAP_m':>12s} {'mAP50-95':>12s} | {'M par':>6s} {'GFLOPs':>7s}")
    for r in rows:
        f = lambda k: f"{r[k]:5.1f}±{r[k + '_std']:.1f}" if r[k + "_std"] == r[k + "_std"] else f"{r[k]:5.1f}     "
        print(f"{r['name']:18s} {r['arch']:8s} {r['imgsz']:5d} {r['attn']:>5s} {r['loss']:>12s} | {f('map50'):>12s} {f('map50_s'):>12s} {f('map50_m'):>12s} {f('map50_95'):>12s} | {r['params_M']:6.2f} {r['gflops']:7.1f}")
        
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("tersimpan:", a.csv)


if __name__ == "__main__":
    main()
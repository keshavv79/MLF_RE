"""Rewrite the no-RIS rows of the fig8 CSVs from the freshly retrained
checkpoints (RIS rows are left untouched)."""
import csv, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent
POWERS = ["35", "37.5", "40", "42.5", "45"]

def load_val(pt):
    d = torch.load(pt, map_location="cpu", weights_only=False)
    v = d["val"]; return v["R"], v["qany"], v["qc"], v["qs"]

def rebuild(csv_path, branch, pt_dir, pt_tmpl):
    rows = list(csv.DictReader(open(csv_path, newline="")))
    keep = [r for r in rows if r["branch"] != branch]           # RIS rows
    new = []
    for P in POWERS:
        R, qany, qc, qs = load_val(ROOT / pt_dir / pt_tmpl.format(P=P))
        new.append(dict(branch=branch, P_dBm=P, batch="128", lr="0.0005",
                        epochs="60", val_R=R, val_qany=qany, val_qc=qc,
                        val_qs=qs, source=pt_tmpl.format(P=P).replace(".pt", "")))
    fields = ["branch","P_dBm","batch","lr","epochs","val_R","val_qany","val_qc","val_qs","source"]
    # keep RIS rows first then no-RIS, sorted as before (no_ris block then ris block)
    out = new + keep
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in out: w.writerow(r)
    print("rewrote", csv_path.name)
    for r in new: print(f"   {branch} P={r['P_dBm']:>5}  R={r['val_R']:.3f}  qany={r['val_qany']:.3f}")

rebuild(ROOT/"_fair_noma"/"fig8"/"fig8_results.csv", "no_ris",
        "_fair_noma/fig8/no_ris", "noris_P{P}.pt")
rebuild(ROOT/"_fair_oma"/"fig8_oma"/"fig8_oma_results.csv", "no_ris_oma",
        "_fair_oma/fig8_oma/no_ris", "noris_oma_P{P}.pt")

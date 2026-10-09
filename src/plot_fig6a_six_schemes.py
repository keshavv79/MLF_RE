"""
plot_fig6a_six_schemes.py
====================================================================
Renders the weighted sum-rate CDF across the six NOMA schemes.
Reads results/csv/fig6_six_scheme_percurve.csv produced by
fig6_six_scheme_compare.py (one column per scheme, each containing
the per-channel weighted objective R_DL_sum = 0.7*(R_n+R_f) + 0.3*R_s).

Output: results/figures/fig6a_six_schemes.png / .pdf
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # .../src
REPO_ROOT = HERE.parent
CSV_IN = REPO_ROOT / "results" / "csv" / "fig6_six_scheme_percurve.csv"
OUT_DIR = REPO_ROOT / "results" / "figures"

# Legend order + styling: Proposed highlighted in red/solid, baselines in
# muted colors, no-RIS reference as a dashed line.
STYLE = {
    "Proposed (RIS+ML optimization)":      dict(color="#d62728", ls="-",  lw=2.6),
    "Equal power + random RIS phases":     dict(color="#1f77b4", ls="-",  lw=1.8),
    "Random power + random RIS phases":    dict(color="#9467bd", ls="-",  lw=1.8),
    "Equal power + fixed RIS phases":      dict(color="#2ca02c", ls="--", lw=1.8),
    "Random power + fixed RIS phases":     dict(color="#8c564b", ls="--", lw=1.8),
    "NOMA + No-RIS":                       dict(color="#555555", ls=":",  lw=2.2),
}


def load_percurve(csv_path):
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        names = next(r)
        cols = {n: [] for n in names}
        for row in r:
            for n, v in zip(names, row):
                if v != "":
                    cols[n].append(float(v))
    return {n: np.array(v) for n, v in cols.items()}


def main():
    data = load_percurve(CSV_IN)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name, style in STYLE.items():
        if name not in data:
            continue
        xs = np.sort(data[name])
        ys = np.linspace(0, 1, xs.size)
        ax.plot(xs, ys, label=name, **style)

    ax.set_xlabel(r"Weighted sum rate $R_{\mathrm{DL}}$ (bits/s/Hz)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Weighted sum-rate CDF across NOMA schemes", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.3, loc="lower right", framealpha=0.92)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig6a_six_schemes.png"
    out_pdf = OUT_DIR / "fig6a_six_schemes.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")


if __name__ == "__main__":
    main()

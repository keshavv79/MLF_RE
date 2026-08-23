"""
make_loss_curves.py
=====================================================================
Render training loss (and sum-rate) convergence curves for EVERY run:
  rows  = {RIS N-sweep, RIS power-sweep, no-RIS power-sweep}
  cols  = {NOMA, OMA}
Each subplot overlays the per-iteration loss of all configs in that group.
Reads the *_iters.npz logs the trainers already save.
Output: plots_fair/loss_curves/  (loss_grid.png + rate_grid.png)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NOMA, OMA = ROOT / "_fair_noma", ROOT / "_fair_oma"
OUT = ROOT / "plots_fair" / "loss_curves"
OUT.mkdir(parents=True, exist_ok=True)

# group -> scheme -> list of (label, npz_path)
def npz(p): return p if p.exists() else None

def _n8(base, sweep, single):
    p = base / sweep
    return p if p.exists() else base / single

GROUPS = {
    "RIS  N-sweep  (P=15 dBm)": {
        "NOMA": [("N=8",  _n8(NOMA, "fig4/fig4_b64_lr1e-4_iters.npz", "fig4/fig4_iters.npz")),
                 ("N=16", NOMA/"fig5/N16_b64_lr1e-4_iters.npz"),
                 ("N=32", NOMA/"fig5/N32_b64_lr1e-4_iters.npz"),
                 ("N=64", NOMA/"fig5/N64_b64_lr1e-4_iters.npz")],
        "OMA":  [("N=8",  _n8(OMA, "fig4/fig4_oma_b64_lr1e-4_iters.npz", "fig4/fig4_oma_iters.npz")),
                 ("N=16", OMA/"fig5_oma/N16_b64_lr1e-4_iters.npz"),
                 ("N=32", OMA/"fig5_oma/N32_b64_lr1e-4_iters.npz"),
                 ("N=64", OMA/"fig5_oma/N64_b64_lr1e-4_iters.npz")],
    },
    "RIS  power-sweep  (N=8)": {
        "NOMA": [(f"P={P}", NOMA/f"fig8/ris/ris_P{P}_b64_lr1e-4_iters.npz")
                 for P in ["5","10","15","20","25"]],
        "OMA":  [(f"P={P}", OMA/f"fig8_oma/ris/ris_oma_P{P}_b64_lr1e-4_iters.npz")
                 for P in ["5","10","15","20","25"]],
    },
    "no-RIS  power-sweep": {
        "NOMA": [(f"P={P}", NOMA/f"fig8/no_ris/noris_P{P}_iters.npz")
                 for P in ["5","10","15","20","25"]],
        "OMA":  [(f"P={P}", OMA/f"fig8_oma/no_ris/noris_oma_P{P}_iters.npz")
                 for P in ["5","10","15","20","25"]],
    },
}


def smooth(x, k=101):
    if len(x) < k: return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def render(metric, fname, ylabel, title):
    rows, cols = list(GROUPS), ["NOMA", "OMA"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 11))
    cmap = plt.get_cmap("viridis")
    for r, gname in enumerate(rows):
        for c, scheme in enumerate(cols):
            ax = axes[r][c]
            entries = GROUPS[gname][scheme]
            present = [(lbl, p) for lbl, p in entries if p and Path(p).exists()]
            if not present:
                ax.text(0.5, 0.5, "no iter log\n(re-run pending)", ha="center",
                        va="center", transform=ax.transAxes, color="#999")
            for i, (lbl, p) in enumerate(present):
                d = np.load(p)
                y = smooth(d[metric])
                ax.plot(y, lw=1.4, color=cmap(i / max(len(present) - 1, 1)), label=lbl)
                ax.legend(fontsize=7, ncol=2)
            ax.set_title(f"{scheme}  |  {gname}", fontsize=9)
            ax.grid(alpha=.3)
            if c == 0: ax.set_ylabel(ylabel)
            if r == len(rows) - 1: ax.set_xlabel("training iteration")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print("wrote", OUT / fname)


if __name__ == "__main__":
    render("iter_losses", "loss_grid.png", "Lagrangian loss",
           "Training loss convergence — all runs")
    render("iter_R", "rate_grid.png", r"$R_{\mathrm{sum}}$ (bps/Hz)",
           "Training sum-rate convergence — all runs")

"""
render_fair_plots.py
====================================================================
Anshul-style paper figures for the M=4 v3-FAIR pipeline, rendered from
the artifacts the training notebook already produces.  Runs on Kaggle
(cwd=/kaggle/working) or locally (--root <dir>).  One scheme per call.

Produces  plots/ :
    fig4_training_loss.png   2-panel meta-loss vs iteration (N-sweep)
    fig5_rsum_vs_N.png       R_sum vs N  (Mbps, RIS / no-RIS / random baseline)
    fig8_rsum_vs_P.png       R_sum vs P_max (Mbps, RIS vs no-RIS)
    loss_curves_grid.png     loss + rate convergence for every run

Usage:  python render_fair_plots.py --scheme noma   (or oma)
"""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BW_MHZ = 10.0    # Mbps conversion (locked, matches Anshul)


def cfg(scheme):
    if scheme == "noma":
        return dict(
            color="#d62728", name="NOMA",
            fig4="fig4/fig4_iters.npz",
            fig4_sweep="fig4/fig4_b{b}_lr{lr}_iters.npz",
            fig5_csv="fig5/fig5_results.csv",
            fig5_iter="fig5/N{N}_b64_lr1e-4_iters.npz",
            fig8_csv="fig8/fig8_results.csv",
            ris_iter="fig8/ris/ris_P{P}_b64_lr1e-4_iters.npz",
            nor_iter="fig8/no_ris/noris_P{P}_iters.npz",
            ris_branch="ris", nor_branch="no_ris",
            mat8="ISAC_RIS_NOMA_channels_v3_fair.mat",
            matN="ISAC_RIS_NOMA_channels_v3_fair_N{N}.mat")
    return dict(
        color="#1f77b4", name="OMA-TDMA",
        fig4="fig4/fig4_oma_iters.npz",
        fig4_sweep="fig4/fig4_oma_b{b}_lr{lr}_iters.npz",
        fig5_csv="fig5_oma/fig5_oma_results.csv",
        fig5_iter="fig5_oma/N{N}_b64_lr1e-4_iters.npz",
        fig8_csv="fig8_oma/fig8_oma_results.csv",
        ris_iter="fig8_oma/ris/ris_oma_P{P}_b64_lr1e-4_iters.npz",
        nor_iter="fig8_oma/no_ris/noris_oma_P{P}_iters.npz",
        ris_branch="ris_oma", nor_branch="no_ris_oma",
        mat8="ISAC_RIS_OMA_channels_v3_fair.mat",
        matN="ISAC_RIS_OMA_channels_v3_fair_N{N}.mat")


def smooth(x, w):
    x = np.asarray(x, float)
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def read_csv(p):
    return list(csv.DictReader(open(p, newline="")))


def baseline_rdl(mat):
    """random-RIS + fixed-alloc weighted sum-rate straight from the .mat."""
    try:
        import h5py
        with h5py.File(mat, "r") as f:
            rn = np.array(f["R_n_all"]).squeeze().mean()
            rf = np.array(f["R_f_all"]).squeeze().mean()
            rs = np.array(f["R_s_all"]).squeeze().mean()
        return 0.7 * (rn + rf) + 0.3 * rs
    except Exception as e:
        print("  baseline read failed:", e); return None


# ----------------------------------------------------------- fig4
def fig4(C, root, out):
    """Anshul-style: batch x lr sweep. batch=colour, lr=line-style."""
    BATCH_COLORS = {32: "#1f77b4", 64: "#d62728", 128: "#2ca02c"}
    LR_STYLES = {"1e-4": "-", "1e-5": "--"}
    configs = [(b, lr) for b in (32, 64, 128) for lr in ("1e-4", "1e-5")]
    present = [(b, lr, root / C["fig4_sweep"].format(b=b, lr=lr))
               for (b, lr) in configs]
    present = [(b, lr, p) for (b, lr, p) in present if p.exists()]
    if not present:                                    # fallback: single fig4 log
        p = root / C["fig4"]
        if p.exists():
            present = [(64, "1e-4", p)]
        else:
            print("fig4: no iter logs"); return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})
    for (b, lr, p) in present:
        L = np.load(p)["iter_losses"].astype(float)
        w = max(1, len(L) // 60)
        Lsm = smooth(L, w); xs = np.arange(len(L))[w - 1:]
        st = dict(color=BATCH_COLORS.get(b, "#333"), linestyle=LR_STYLES.get(lr, "-"),
                  lw=2, alpha=0.95)
        axA.plot(xs, Lsm, label=f"batch={b}, lr={lr}", **st)
        keep = xs >= 200
        axB.plot(xs[keep], Lsm[keep], **st)
    axA.set_xlabel("training iteration"); axA.set_ylabel(r"meta-loss  $L=-\mathbb{E}[R_{sum}]+$ penalties")
    axA.set_title("(a) full trajectory"); axA.grid(alpha=.3); axA.legend(fontsize=8, ncol=2)
    axB.set_xlabel("training iteration"); axB.set_title(r"(b) steady-state (iter $\geq$ 200)")
    axB.grid(alpha=.3)
    fig.suptitle(f"Fig 4 — {C['name']} MAML training loss vs iterations "
                 f"(batch x lr sweep, M=4 fair, N=8)", fontsize=12, y=1.02)
    plt.tight_layout(); plt.savefig(out / "fig4_training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(); print(f"saved fig4_training_loss.png  ({len(present)} configs)")


# ----------------------------------------------------------- fig5
def fig5(C, root, out):
    csvp = root / C["fig5_csv"]
    if not csvp.exists():
        print("fig5: no csv"); return
    rows = sorted(read_csv(csvp), key=lambda r: int(r["N"]))
    Ns = [int(r["N"]) for r in rows]
    R_dl = [float(r["val_R"]) * BW_MHZ for r in rows]
    # random-RIS baseline per N (from the .mat)
    R_base = []
    for N in Ns:
        mat = root / (C["mat8"] if N == 8 else C["matN"].format(N=N))
        b = baseline_rdl(mat) if mat.exists() else None
        R_base.append(b * BW_MHZ if b else np.nan)
    # no-RIS reference at P=40 (from fig8 csv)
    R_nor = None
    f8 = root / C["fig8_csv"]
    if f8.exists():
        for r in read_csv(f8):
            if r["branch"] == C["nor_branch"] and abs(float(r["P_dBm"]) - 40) < 1e-6:
                R_nor = float(r["val_R"]) * BW_MHZ
    col = C["color"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(Ns, R_dl, color=col, ls="-", lw=2.2, marker="v", ms=10, mfc="white",
            mec=col, mew=1.8, label=f"{C['name']}-RIS (MAML-DL, joint $\\Theta,\\alpha$)")
    if R_nor is not None:
        ax.plot(Ns, [R_nor] * len(Ns), color=col, ls="--", lw=2, marker="^", ms=10,
                mfc="white", mec=col, mew=1.8, label=f"{C['name']} no-RIS (P=40) — {R_nor:.1f} Mbps")
        im = len(Ns) // 2
        ax.annotate("", xy=(Ns[im], R_dl[im]), xytext=(Ns[im], R_nor),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        ax.text(Ns[im] * 1.04, (R_nor + R_dl[im]) / 2, "RIS gain", fontsize=10, va="center")
    if not all(np.isnan(R_base)):
        ax.plot(Ns, R_base, color="#888", ls=":", lw=1.5, marker="s", ms=7,
                label=f"{C['name']}-RIS baseline (random $\\Theta$, fixed $\\alpha$)")
    for x, y in zip(Ns, R_dl):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=col)
    ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of RIS elements, $N$", fontsize=12)
    ax.set_ylabel(f"Weighted sum rate (Mbps, BW={BW_MHZ:g} MHz)", fontsize=12)
    ax.set_title(f"Fig 5 — $R_{{DL,sum}}$ vs N  ({C['name']}, M=4 fair, P=40 dBm)", fontsize=11)
    ax.grid(alpha=.3, which="both"); ax.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout(); plt.savefig(out / "fig5_rsum_vs_N.png", dpi=200, bbox_inches="tight")
    plt.close(); print("saved fig5_rsum_vs_N.png")


# ----------------------------------------------------------- fig8
def fig8(C, root, out):
    csvp = root / C["fig8_csv"]
    if not csvp.exists():
        print("fig8: no csv"); return
    rows = read_csv(csvp)

    def branch(b):
        rs = sorted([r for r in rows if r["branch"] == b], key=lambda r: float(r["P_dBm"]))
        return [float(r["P_dBm"]) for r in rs], [float(r["val_R"]) * BW_MHZ for r in rs]

    Pr, Rr = branch(C["ris_branch"]); Pn, Rn = branch(C["nor_branch"])
    col = C["color"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(Pr, Rr, color=col, ls="-", lw=2.2, marker="D", ms=9, mfc="white", mec=col,
            mew=1.8, label=f"{C['name']}-RIS, DL (MAML)")
    ax.plot(Pn, Rn, color=col, ls="--", lw=2, marker="s", ms=9, mfc="white", mec=col,
            mew=1.8, label=f"{C['name']} no-RIS, DL (single-shot)")
    if Pr and Pn:
        im = len(Pr) // 2
        ax.annotate("", xy=(Pr[im], Rr[im]), xytext=(Pr[im], Rn[im]),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        ax.text(Pr[im] + 0.25, (Rn[im] + Rr[im]) / 2, "RIS gain", fontsize=10, va="center")
    for x, y in zip(Pr, Rr):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=col)
    for x, y in zip(Pn, Rn):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=9, color=col)
    ax.set_xlabel("Transmit power $P_{max}$ (dBm)", fontsize=12)
    ax.set_ylabel(f"Weighted sum rate (Mbps, BW={BW_MHZ:g} MHz)", fontsize=12)
    ax.set_title(f"Fig 8 — $R_{{DL,sum}}$ vs $P_{{max}}$  ({C['name']}, M=4 fair, N=8)", fontsize=11)
    ax.set_xticks(Pr); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout(); plt.savefig(out / "fig8_rsum_vs_P.png", dpi=200, bbox_inches="tight")
    plt.close(); print("saved fig8_rsum_vs_P.png")


# ----------------------------------------------------- loss grid
def loss_grid(C, root, out):
    n8 = root / C["fig4_sweep"].format(b=64, lr="1e-4")
    if not n8.exists():
        n8 = root / C["fig4"]
    groups = [
        ("RIS N-sweep", [("N=8", n8)] +
         [(f"N={N}", root / C["fig5_iter"].format(N=N)) for N in (16, 32, 64)]),
        ("RIS power-sweep", [(f"P={P}", root / C["ris_iter"].format(P=P))
                             for P in ["5", "10", "15", "20", "25"]]),
        ("no-RIS power-sweep", [(f"P={P}", root / C["nor_iter"].format(P=P))
                                for P in ["5", "10", "15", "20", "25"]]),
    ]
    for metric, fname, ylab in [("iter_losses", "loss_curves_grid.png", "Lagrangian loss"),
                                ("iter_R", "rate_curves_grid.png", r"$R_{sum}$ (bps/Hz)")]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        cmap = plt.get_cmap("viridis")
        for ax, (gname, entries) in zip(axes, groups):
            present = [(l, p) for l, p in entries if p.exists()]
            for i, (lbl, p) in enumerate(present):
                ax.plot(smooth(np.load(p)[metric], 101), lw=1.4,
                        color=cmap(i / max(len(present) - 1, 1)), label=lbl)
            if present:
                ax.legend(fontsize=7, ncol=2)
            else:
                ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999")
            ax.set_title(f"{C['name']} | {gname}", fontsize=9); ax.grid(alpha=.3)
            ax.set_xlabel("iteration"); ax.set_ylabel(ylab)
        fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close()
        print("saved", fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["noma", "oma"], required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="plots")
    a = ap.parse_args()
    C = cfg(a.scheme)
    root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    print(f"=== rendering {C['name']} plots -> {out} ===")
    fig4(C, root, out); fig5(C, root, out); fig8(C, root, out); loss_grid(C, root, out)
    print("done:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()

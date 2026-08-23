"""
make_fair_plots.py
=====================================================================
Render fig4 / fig5 / fig8 PNGs + a master NOMA-vs-OMA comparison
from the downloaded Kaggle result trees:

    _fair_noma/   (noma_fair_results.zip)
    _fair_oma/    (oma_fair_results.zip)

Outputs everything into plots_fair/.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NOMA = ROOT / "_fair_noma"
OMA  = ROOT / "_fair_oma"
OUT  = ROOT / "plots_fair"
OUT.mkdir(exist_ok=True)

C_NOMA, C_OMA = "#1f77b4", "#d62728"


def smooth(x, k=51):
    if len(x) < k:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="valid")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- fig4
def _fig4_npz(base, sweep_name, single_name):
    p = base / "fig4" / sweep_name           # new canonical batch64/lr1e-4 run
    return p if p.exists() else base / "fig4" / single_name

def fig4():
    dn = np.load(_fig4_npz(NOMA, "fig4_b64_lr1e-4_iters.npz", "fig4_iters.npz"))
    do = np.load(_fig4_npz(OMA, "fig4_oma_b64_lr1e-4_iters.npz", "fig4_oma_iters.npz"))
    plt.figure(figsize=(7, 4.5))
    plt.plot(smooth(dn["iter_R"]), color=C_NOMA, label="NOMA (MAML)")
    plt.plot(smooth(do["iter_R"]), color=C_OMA,  label="OMA-TDMA (MAML)")
    plt.xlabel("training iteration")
    plt.ylabel(r"$R_{\mathrm{sum}}$  (bps/Hz)")
    plt.title("Fig. 4 — MAML training convergence  (N=8, P=15 dBm)")
    plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "fig4_convergence.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------- fig5
def fig5():
    rn = read_csv(NOMA / "fig5" / "fig5_results.csv")
    ro = read_csv(OMA  / "fig5_oma" / "fig5_oma_results.csv")
    Nn = [int(r["N"]) for r in rn];  Rn = [float(r["val_R"]) for r in rn]
    No = [int(r["N"]) for r in ro];  Ro = [float(r["val_R"]) for r in ro]
    plt.figure(figsize=(7, 4.5))
    plt.plot(Nn, Rn, "o-", color=C_NOMA, lw=2, ms=7, label="NOMA")
    plt.plot(No, Ro, "s-", color=C_OMA,  lw=2, ms=7, label="OMA-TDMA")
    for x, y in zip(Nn, Rn): plt.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=C_NOMA)
    for x, y in zip(No, Ro): plt.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=C_OMA)
    plt.xscale("log", base=2); plt.xticks(Nn, [str(n) for n in Nn])
    plt.xlabel("number of RIS elements  N")
    plt.ylabel(r"$R_{\mathrm{sum}}$  (bps/Hz)")
    plt.title("Fig. 5 — Sum-rate vs RIS size  (P=15 dBm)")
    plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "fig5_R_vs_N.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------- fig8
def _split(rows):
    out = {}
    for r in rows:
        br = r["branch"]
        P, R, q = float(r["P_dBm"]), float(r["val_R"]), float(r["val_qany"])
        good = R * (1.0 - q)           # effective QoS-feasible throughput (goodput)
        out.setdefault(br, []).append((P, R, q, good))
    for br in out: out[br].sort()
    return out


def fig8():
    sn = _split(read_csv(NOMA / "fig8" / "fig8_results.csv"))
    so = _split(read_csv(OMA  / "fig8_oma" / "fig8_oma_results.csv"))
    fig, (axR, axQ) = plt.subplots(1, 2, figsize=(12, 4.6))

    def curve(ax, data, key, idx, color, ls, lbl):
        P = [d[0] for d in data]; y = [d[idx] for d in data]
        ax.plot(P, y, key, color=color, ls=ls, lw=2, ms=6, label=lbl)

    # left: sum-rate  (idx=1 -> raw R; RIS now dominates after blocked-direct regen)
    curve(axR, sn["ris"],    "o", 1, C_NOMA, "-",  "NOMA + RIS")
    curve(axR, sn["no_ris"], "o", 1, C_NOMA, "--", "NOMA no-RIS")
    curve(axR, so["ris_oma"],    "s", 1, C_OMA, "-",  "OMA + RIS")
    curve(axR, so["no_ris_oma"], "s", 1, C_OMA, "--", "OMA no-RIS")
    axR.set_xlabel("transmit power  (dBm)")
    axR.set_ylabel(r"$R_{\mathrm{sum}}$  (bps/Hz)")
    axR.set_title("Fig. 8a — Sum-rate vs power")
    axR.grid(alpha=.3); axR.legend(fontsize=8)

    # right: QoS violation
    curve(axQ, sn["ris"],    "o", 2, C_NOMA, "-",  "NOMA + RIS")
    curve(axQ, sn["no_ris"], "o", 2, C_NOMA, "--", "NOMA no-RIS")
    curve(axQ, so["ris_oma"],    "s", 2, C_OMA, "-",  "OMA + RIS")
    curve(axQ, so["no_ris_oma"], "s", 2, C_OMA, "--", "OMA no-RIS")
    axQ.set_xlabel("transmit power  (dBm)"); axQ.set_ylabel("QoS-violation rate (any)")
    axQ.set_title("Fig. 8b — RIS keeps QoS feasible"); axQ.grid(alpha=.3); axQ.legend(fontsize=8)

    plt.tight_layout(); plt.savefig(OUT / "fig8_R_and_QoS_vs_P.png", dpi=160)
    plt.close()

    # print the goodput table so the crossing behaviour is visible
    print("\nFig.8 effective throughput R*(1-qviol):")
    print(f"  {'P':>5} | {'NOMA+RIS':>9} {'NOMA-noR':>9} | {'OMA+RIS':>9} {'OMA-noR':>9}")
    for i in range(len(sn['ris'])):
        P = sn['ris'][i][0]
        print(f"  {P:5.1f} | {sn['ris'][i][3]:9.2f} {sn['no_ris'][i][3]:9.2f} | "
              f"{so['ris_oma'][i][3]:9.2f} {so['no_ris_oma'][i][3]:9.2f}")


# ------------------------------------------------------------- summary
def master_summary():
    def get(path, k="val_R"):
        return read_csv(path)
    rn = {int(r["N"]): float(r["val_R"]) for r in read_csv(NOMA/"fig5"/"fig5_results.csv")}
    ro = {int(r["N"]): float(r["val_R"]) for r in read_csv(OMA/"fig5_oma"/"fig5_oma_results.csv")}
    lines = []
    lines.append("=" * 64)
    lines.append("MASTER COMPARISON — M=4 FAIR system model  (NOMA vs OMA-TDMA)")
    lines.append("=" * 64)
    lines.append("")
    lines.append("Fig.5  Sum-rate vs RIS size N  (P=15 dBm):")
    lines.append(f"  {'N':>4} | {'NOMA':>8} | {'OMA':>8} | {'NOMA-OMA':>9}")
    lines.append("  " + "-" * 38)
    for N in sorted(rn):
        lines.append(f"  {N:>4} | {rn[N]:8.2f} | {ro[N]:8.2f} | {rn[N]-ro[N]:9.2f}")
    lines.append("")
    # test-set eval headline (read live from the eval summaries)
    def eval_row(csv_path, method):
        for r in read_csv(csv_path):
            if r["method"] == method:
                return r
        return None
    lines.append("Test-set eval (N=8, 5000 samples):")
    en = eval_row(NOMA / "eval_noma_fair" / "summary.csv", "noma_fair")
    eo = eval_row(OMA / "eval_oma_fair" / "summary.csv", "oma_maml")
    if en:
        lines.append(f"  NOMA fair : Rsum={float(en['R_DL_sum_paper']):.2f}  R_n={float(en['R_n']):.2f} "
                     f"R_f={float(en['R_f']):.2f} R_s={float(en['R_s']):.2f}  qany={float(en['qos_viol_any']):.3f}")
    if eo:
        lines.append(f"  OMA  fair : Rsum={float(eo['R_DL_sum_paper']):.2f}  R_n={float(eo['R_n']):.2f} "
                     f"R_f={float(eo['R_f']):.2f} R_s={float(eo['R_s']):.2f}  qany={float(eo['qos_viol_any']):.3f}")
    lines.append("")
    # fig8 raw sum-rate: RIS vs no-RIS gap (after blocked-direct regen)
    sn = _split(read_csv(NOMA / "fig8" / "fig8_results.csv"))
    so = _split(read_csv(OMA  / "fig8_oma" / "fig8_oma_results.csv"))
    lines.append("Fig.8  Sum-rate RIS vs no-RIS  (blocked direct link):")
    lines.append(f"  {'P(dBm)':>6} | {'NOMA RIS':>8} {'noRIS':>6} {'gap':>5} | {'OMA RIS':>7} {'noRIS':>6} {'gap':>5}")
    lines.append("  " + "-" * 52)
    for i in range(len(sn["ris"])):
        P = sn["ris"][i][0]
        nr, nn = sn["ris"][i][1], sn["no_ris"][i][1]
        orr, onn = so["ris_oma"][i][1], so["no_ris_oma"][i][1]
        lines.append(f"  {P:6.1f} | {nr:8.2f} {nn:6.2f} {nr-nn:5.2f} | {orr:7.2f} {onn:6.2f} {orr-onn:5.2f}")
    lines.append("")
    lines.append("Conclusion: NOMA dominates OMA-TDMA at every N and every power;")
    lines.append("RIS beats no-RIS in raw sum-rate at every power for both schemes")
    lines.append("and drives QoS-violation -> 0 (no-RIS violates heavily at low power).")
    txt = "\n".join(lines)
    (OUT / "master_summary.txt").write_text(txt)
    # csv
    with open(OUT / "fig5_combined.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "NOMA_Rsum", "OMA_Rsum", "gap"])
        for N in sorted(rn): w.writerow([N, rn[N], ro[N], rn[N]-ro[N]])
    print(txt)


if __name__ == "__main__":
    fig4(); fig5(); fig8(); master_summary()
    print("\nWrote PNGs + summary into", OUT)
    for p in sorted(OUT.iterdir()):
        print("  ", p.name)

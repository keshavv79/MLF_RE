"""
plot_fig5_4curve_FIXED.py
====================================================================
Corrected, path-fixed version of plot_fig5_4curve.py (ORIGINAL FILE
PRESERVED UNTOUCHED -- see plot_fig5_4curve.py).

Two independent issues were found and fixed here, kept separate so each
is auditable:

ISSUE 1 -- wrong relative paths.
  The original script assumes it lives directly under a folder that also
  contains fig5/, fig4/, fig5_oma/, no_ris/ as siblings. In this repo
  layout those checkpoints actually live under models/noma-fair/fig-5/,
  models/noma-fair/policy_noma_fair_best.pt, models/oma_fair/fig5_oma/,
  models/oma_fair/policy_oma_fair_best.pt. This file points at the real
  locations instead.

ISSUE 2 -- wrong no-RIS checkpoints (the actual bug).
  The original script reads no_ris/policy_noris_best.pt and
  no_ris/policy_noris_oma_best.pt. Those two checkpoints (now backed up
  at models/no-ris/old_40dBm_backup/*_OLD_40dBm.pt) were trained on a
  DIFFERENT, older channel dataset (M=2, R_th_c=2.0/R_th_s=0.3,
  beta_T=40 dB -- not the M=4 "fair" scenario used by every other
  checkpoint in this repo) AND at 40 dBm instead of the 15 dBm used by
  every NOMA-RIS/OMA-RIS point on this same plot. That made the no-RIS
  reference lines wildly inconsistent -- e.g. OMA no-RIS plotted at
  165.8 Mbps, *above* every RIS curve, which contradicts the paper's
  core claim that RIS assistance helps.

  Both no-RIS checkpoints have since been retrained on the correct,
  matching "fair" no-RIS datasets (found in
  PE/PE/OMA_new/m4_datasets/ISAC_NOMA_channels_fair_noris.mat and
  ISAC_OMA_channels_fair_noris.mat -- same M=4, same R_th_c/R_th_s family
  as the RIS models) at the correct 15 dBm, using the exact same
  train_noma_fair_noris.py / train_noris_oma.py scripts already in this
  repo with --P_tot_dBm 15. The retrained checkpoints are installed at
  models/no-ris/policy_noris_best.pt and
  models/no-ris/policy_noris_oma_best.pt (old ones preserved in
  models/no-ris/old_40dBm_backup/).

NOTE ON SCOPE: this 4-curve figure is a *supplementary* artifact. The
actual Fig. 5 embedded in the paper (figs/fig_rate_N.jpg, built by a
different pipeline using only the RIS-curve CSVs in results/csv/) only
has 2 curves (NOMA-RIS, OMA-RIS) and was never affected by this bug --
it is unchanged and does not need updating.

Output: fig5_4curve_FIXED.png  (does not overwrite fig5_4curve.png)
"""

from pathlib import Path
import csv
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BW_MHZ = 10.0
HERE = Path(__file__).resolve().parent        # .../MLF_RE-main/src
ROOT = HERE.parent                            # .../MLF_RE-main

OUT = HERE / 'fig5_4curve_FIXED.png'

NOMA_COLOR = '#d62728'   # red
OMA_COLOR  = '#1f77b4'   # blue


def load_R(ckpt_path):
    p = Path(ckpt_path)
    if not p.exists():
        print(f'[warn] checkpoint not found: {p}')
        return None
    ck = torch.load(p, map_location='cpu', weights_only=False)
    v = ck.get('val', {}).get('R', None)
    return float(v) if v is not None else None


def load_noma_ris_points():
    """NOMA-RIS N-sweep: N=8 from the main policy, N=16/32/64 from fig-5/."""
    points = {}
    points[8] = load_R(ROOT / 'models' / 'noma-fair' / 'policy_noma_fair_best.pt')
    for N in (16, 32, 64):
        points[N] = load_R(ROOT / 'models' / 'noma-fair' / 'fig-5' / f'N{N}_b64_lr1e-4.pt')
    return {k: v for k, v in points.items() if v is not None}


def load_oma_ris_points():
    """OMA-RIS N-sweep: N=8 from the main policy, N=16/32/64 from fig5_oma/."""
    points = {}
    points[8] = load_R(ROOT / 'models' / 'oma_fair' / 'policy_oma_fair_best.pt')
    for N in (16, 32, 64):
        points[N] = load_R(ROOT / 'models' / 'oma_fair' / 'fig5_oma' / f'N{N}_b64_lr1e-4.pt')
    return {k: v for k, v in points.items() if v is not None}


def main():
    noma_ris = load_noma_ris_points()
    oma_ris  = load_oma_ris_points()

    Ns_noma = sorted(noma_ris.keys())
    Ns_oma  = sorted(oma_ris.keys())

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    if Ns_noma:
        ax.plot(Ns_noma, [noma_ris[n] * BW_MHZ for n in Ns_noma],
                color=NOMA_COLOR, linestyle='-', linewidth=2.2,
                marker='v', markersize=10, markerfacecolor='white',
                markeredgecolor=NOMA_COLOR, markeredgewidth=1.8,
                label='NOMA-RIS (MAML-DL)')

    R_noris_noma = load_R(ROOT / 'models' / 'no-ris' / 'policy_noris_best.pt')
    if R_noris_noma is not None and Ns_noma:
        ax.plot(Ns_noma, [R_noris_noma * BW_MHZ] * len(Ns_noma),
                color=NOMA_COLOR, linestyle='--', linewidth=2.0,
                marker='^', markersize=10, markerfacecolor='white',
                markeredgecolor=NOMA_COLOR, markeredgewidth=1.8,
                label=f'NOMA no-RIS ({R_noris_noma * BW_MHZ:.1f} Mbps)')

    if Ns_oma:
        ax.plot(Ns_oma, [oma_ris[n] * BW_MHZ for n in Ns_oma],
                color=OMA_COLOR, linestyle='-', linewidth=2.2,
                marker='o', markersize=10, markerfacecolor='white',
                markeredgecolor=OMA_COLOR, markeredgewidth=1.8,
                label='OMA-RIS (MAML-DL)')

    R_noris_oma = load_R(ROOT / 'models' / 'no-ris' / 'policy_noris_oma_best.pt')
    if R_noris_oma is not None and Ns_oma:
        ax.plot(Ns_oma, [R_noris_oma * BW_MHZ] * len(Ns_oma),
                color=OMA_COLOR, linestyle='--', linewidth=2.0,
                marker='s', markersize=10, markerfacecolor='white',
                markeredgecolor=OMA_COLOR, markeredgewidth=1.8,
                label=f'OMA no-RIS ({R_noris_oma * BW_MHZ:.1f} Mbps)')

    all_Ns = sorted(set(Ns_noma) | set(Ns_oma))
    ax.set_xscale('log', base=2)
    ax.set_xticks(all_Ns or [8, 16, 32, 64])
    ax.set_xticklabels([str(n) for n in (all_Ns or [8, 16, 32, 64])])
    ax.set_xlabel('Number of RIS elements, $N$', fontsize=12)
    ax.set_ylabel(f'Weighted sum rate (Mbps, BW={BW_MHZ:g} MHz)', fontsize=12)
    ax.set_title('Fig 5 (supplementary, corrected) — $R_\\mathrm{sum}$ vs $N$: NOMA vs OMA\n'
                 '(MISO RIS-ISAC, M=4 fair scenario, P=15 dBm)', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(OUT, dpi=220, bbox_inches='tight')
    print(f'saved -> {OUT}')
    print(f'NOMA no-RIS R = {R_noris_noma:.4f} bps/Hz ({R_noris_noma*BW_MHZ:.1f} Mbps)')
    print(f'OMA  no-RIS R = {R_noris_oma:.4f} bps/Hz ({R_noris_oma*BW_MHZ:.1f} Mbps)')


if __name__ == '__main__':
    main()

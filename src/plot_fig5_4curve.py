"""
plot_fig5_4curve.py
====================================================================
Fig 5: weighted sum rate vs N (RIS elements) — 4 curves.
  NOMA-RIS, NOMA no-RIS, OMA-RIS, OMA no-RIS.

Reads NOMA results directly from .pt checkpoints in fig5/ and no_ris/.
Reads OMA  results from fig5_oma/fig5_oma_results.csv (run aggregate first).
"""

from pathlib import Path
import csv
import torch
import matplotlib.pyplot as plt

BW_MHZ = 10.0
HERE = Path(__file__).resolve().parent   # plots/
ROOT = HERE.parent                       # OMA_new/

OUT = HERE / 'fig5_4curve.png'

NOMA_COLOR = '#d62728'   # red
OMA_COLOR  = '#1f77b4'   # blue


def load_noma_ris_points():
    """Load NOMA-RIS R values directly from .pt checkpoints."""
    points = {}
    # N=16, 32, 64
    for N in (16, 32, 64):
        p = ROOT / 'fig5' / f'N{N}_b64_lr1e-4.pt'
        if p.exists():
            ck = torch.load(p, map_location='cpu', weights_only=False)
            points[N] = float(ck.get('val', {}).get('R', 0))
        else:
            print(f'[warn] NOMA N={N} checkpoint not found: {p}')
    # N=8: look in fig4/ (user may need to copy from Anshul's folder)
    p8 = ROOT / 'fig4' / 'fig4_b64_lr1e-4.pt'
    if p8.exists():
        ck = torch.load(p8, map_location='cpu', weights_only=False)
        points[8] = float(ck.get('val', {}).get('R', 0))
    else:
        print(f'[warn] NOMA N=8 checkpoint not found: {p8}')
        print('       Copy fig4_b64_lr1e-4.pt from Anshul folder/fig4/ into OMA_new/fig4/')
    return points


def load_noris_R(ckpt_path):
    p = Path(ckpt_path)
    if not p.exists():
        return None
    ck = torch.load(p, map_location='cpu', weights_only=False)
    v  = ck.get('val', {}).get('R', None)
    return float(v) if v is not None else None


def load_oma_ris_points():
    """Load OMA-RIS R values from fig5_oma_results.csv."""
    csv_path = ROOT / 'fig5_oma' / 'fig5_oma_results.csv'
    if not csv_path.exists():
        print(f'[warn] OMA CSV not found: {csv_path}')
        print('       Run fig5_oma/run_fig5_oma_sweep.ps1 first.')
        return {}
    rows = list(csv.DictReader(open(csv_path)))
    return {int(r['N']): float(r['val_R']) for r in rows}


def main():
    noma_ris = load_noma_ris_points()
    oma_ris  = load_oma_ris_points()

    # Use whatever Ns have both curves (or plot independently)
    Ns_noma = sorted(noma_ris.keys())
    Ns_oma  = sorted(oma_ris.keys())

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    if Ns_noma:
        ax.plot(Ns_noma, [noma_ris[n] * BW_MHZ for n in Ns_noma],
                color=NOMA_COLOR, linestyle='-', linewidth=2.2,
                marker='v', markersize=10, markerfacecolor='white',
                markeredgecolor=NOMA_COLOR, markeredgewidth=1.8,
                label='NOMA-RIS (MAML-DL)')

    R_noris_noma = load_noris_R(ROOT / 'no_ris' / 'policy_noris_best.pt')
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

    R_noris_oma = load_noris_R(ROOT / 'no_ris' / 'policy_noris_oma_best.pt')
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
    ax.set_title('Fig 5 — $R_\\mathrm{sum}$ vs $N$: NOMA vs OMA\n'
                 '(MISO RIS-ISAC, M=2, MAML, b=64, lr=$10^{-4}$, 30 ep)', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(OUT, dpi=220, bbox_inches='tight')
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()

"""
plot_fig8_4curve.py
====================================================================
Fig 8: weighted sum rate vs P_max (dBm) — 4 curves.
  NOMA-RIS, NOMA no-RIS, OMA-RIS, OMA no-RIS.

Reads NOMA results from .pt checkpoints in fig8/ris/ and fig8/no_ris/.
Reads OMA  results from fig8_oma/fig8_oma_results.csv (run aggregate first).
"""

from pathlib import Path
import csv
import torch
import matplotlib.pyplot as plt

BW_MHZ = 10.0
HERE = Path(__file__).resolve().parent   # plots/
ROOT = HERE.parent                       # OMA_new/

OUT = HERE / 'fig8_4curve.png'

NOMA_COLOR = '#d62728'   # red
OMA_COLOR  = '#1f77b4'   # blue

P_TAGS = ['35', '37.5', '40', '42.5', '45']


def load_noma_fig8():
    """Load NOMA fig8 R values directly from .pt checkpoints."""
    ris, nor = {}, {}
    for P in P_TAGS:
        p_ris = ROOT / 'fig8' / 'ris' / f'ris_P{P}_b64_lr1e-4.pt'
        p_nor = ROOT / 'fig8' / 'no_ris' / f'noris_P{P}.pt'
        if p_ris.exists():
            ck = torch.load(p_ris, map_location='cpu', weights_only=False)
            ris[float(P)] = float(ck.get('val', {}).get('R', 0))
        if p_nor.exists():
            ck = torch.load(p_nor, map_location='cpu', weights_only=False)
            nor[float(P)] = float(ck.get('val', {}).get('R', 0))
    return ris, nor


def load_oma_fig8():
    """Load OMA fig8 R values from fig8_oma_results.csv."""
    csv_path = ROOT / 'fig8_oma' / 'fig8_oma_results.csv'
    if not csv_path.exists():
        print(f'[warn] OMA CSV not found: {csv_path}')
        print('       Run fig8_oma/run_fig8_oma_sweep.ps1 first.')
        return {}, {}
    ris, nor = {}, {}
    for r in csv.DictReader(open(csv_path)):
        P = float(r['P_dBm'])
        if r['branch'] == 'ris_oma':
            ris[P] = float(r['val_R'])
        elif r['branch'] == 'no_ris_oma':
            nor[P] = float(r['val_R'])
    return ris, nor


def main():
    noma_ris, noma_nor = load_noma_fig8()
    oma_ris,  oma_nor  = load_oma_fig8()

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    def plot_branch(data, color, ls, marker, label):
        if not data:
            return
        pts = sorted(data.items())
        Ps  = [p for p, _ in pts]
        Rs  = [r * BW_MHZ for _, r in pts]
        ax.plot(Ps, Rs, color=color, linestyle=ls, linewidth=2.2,
                marker=marker, markersize=10,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.8,
                label=label)

    plot_branch(noma_ris, NOMA_COLOR, '-',  'v', 'NOMA-RIS (N=16, MAML-DL)')
    plot_branch(noma_nor, NOMA_COLOR, '--', '^', 'NOMA no-RIS')
    plot_branch(oma_ris,  OMA_COLOR,  '-',  'o', 'OMA-RIS  (N=16, MAML-DL)')
    plot_branch(oma_nor,  OMA_COLOR,  '--', 's', 'OMA no-RIS')

    ax.set_xlabel('Maximum transmit power, $P_{\\max}$ (dBm)', fontsize=12)
    ax.set_ylabel(f'Weighted sum rate (Mbps, BW={BW_MHZ:g} MHz)', fontsize=12)
    ax.set_title('Fig 8 — $R_\\mathrm{sum}$ vs $P_{{\\max}}$: NOMA vs OMA\n'
                 '(MISO RIS-ISAC, N=16, M=2, MAML, b=64, lr=$10^{{-4}}$, 30 ep)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(OUT, dpi=220, bbox_inches='tight')
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()

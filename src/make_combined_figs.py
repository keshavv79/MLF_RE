"""
make_combined_figs.py
====================================================================
Build fig4 / fig5 / fig8 from the downloaded Kaggle results.

Reads (relative to --base, default ../results_attempt2):
  NOMA: noma_results/{policy_maml_ws03.pt, N{16,32,64}_b64_lr1e-4.pt,
                      fig8_results.csv, fig4_iters.npz}
  OMA : oma_results/{fig5_oma/fig5_oma_results.csv,
                     fig8_oma/fig8_oma_results.csv, fig4/fig4_oma_iters.npz}

Outputs (into --base/figures):
  fig5_4curve.png   Rsum vs N   (NOMA/OMA, RIS/no-RIS)
  fig8_4curve.png   Rsum vs P   (NOMA/OMA, RIS/no-RIS)
  fig4_loss.png     training loss curves (NOMA vs OMA)
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BW_MHZ = 10.0
NOMA_C = '#d62728'   # red
OMA_C  = '#1f77b4'   # blue


def pt_R(p):
    p = Path(p)
    if not p.exists():
        print(f'[warn] missing {p}'); return None
    ck = torch.load(p, map_location='cpu', weights_only=False)
    return float(ck.get('val', {}).get('R', 0))


def csv_rows(p):
    p = Path(p)
    if not p.exists():
        print(f'[warn] missing {p}'); return []
    return list(csv.DictReader(open(p)))


def fig5(noma_dir, oma_dir, out):
    # NOMA-RIS from .pt (N=8 is the policy_maml_ws03 checkpoint)
    noma_ris = {8: pt_R(noma_dir/'policy_maml_ws03.pt')}
    for N in (16, 32, 64):
        noma_ris[N] = pt_R(noma_dir/f'N{N}_b64_lr1e-4.pt')
    # OMA-RIS from CSV
    oma_ris = {int(r['N']): float(r['val_R'])
               for r in csv_rows(oma_dir/'fig5_oma'/'fig5_oma_results.csv')}
    # no-RIS reference = value at operating power P=40 dBm from fig8 CSVs
    def noris_at40(rows, branch):
        for r in rows:
            if r['branch'] == branch and abs(float(r['P_dBm']) - 40.0) < 1e-6:
                return float(r['val_R'])
        return None
    noma_noris = noris_at40(csv_rows(noma_dir/'fig8_results.csv'), 'no_ris')
    oma_noris  = noris_at40(csv_rows(oma_dir/'fig8_oma'/'fig8_oma_results.csv'), 'no_ris_oma')

    Ns_n = sorted(noma_ris); Ns_o = sorted(oma_ris)
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    ax.plot(Ns_n, [noma_ris[n]*BW_MHZ for n in Ns_n], color=NOMA_C, lw=2.2,
            marker='v', ms=9, mfc='white', mec=NOMA_C, mew=1.8, label='NOMA-RIS (MAML-DL)')
    if noma_noris is not None:
        ax.plot(Ns_n, [noma_noris*BW_MHZ]*len(Ns_n), color=NOMA_C, ls='--', lw=2,
                label=f'NOMA no-RIS ({noma_noris*BW_MHZ:.0f} Mbps)')
    ax.plot(Ns_o, [oma_ris[n]*BW_MHZ for n in Ns_o], color=OMA_C, lw=2.2,
            marker='o', ms=9, mfc='white', mec=OMA_C, mew=1.8, label='OMA-RIS (MAML-DL)')
    if oma_noris is not None:
        ax.plot(Ns_o, [oma_noris*BW_MHZ]*len(Ns_o), color=OMA_C, ls='--', lw=2,
                label=f'OMA no-RIS ({oma_noris*BW_MHZ:.0f} Mbps)')
    allN = sorted(set(Ns_n) | set(Ns_o)) or [8, 16, 32, 64]
    ax.set_xscale('log', base=2); ax.set_xticks(allN)
    ax.set_xticklabels([str(n) for n in allN])
    ax.set_xlabel('Number of RIS elements, $N$', fontsize=12)
    ax.set_ylabel(f'Weighted sum rate (Mbps, BW={BW_MHZ:g} MHz)', fontsize=12)
    ax.set_title('Fig 5 — $R_\\mathrm{sum}$ vs $N$ : NOMA vs OMA  (MISO RIS-ISAC, M=2)', fontsize=11)
    ax.grid(True, alpha=0.3, which='both'); ax.legend(loc='upper left', fontsize=10, framealpha=0.92)
    plt.tight_layout(); plt.savefig(out, dpi=220, bbox_inches='tight'); plt.close()
    print(f'saved -> {out}')


def fig8(noma_dir, oma_dir, out):
    def series(rows, branch):
        pts = [(float(r['P_dBm']), float(r['val_R'])) for r in rows if r['branch'] == branch]
        pts.sort(); return [p for p, _ in pts], [v for _, v in pts]
    nr = csv_rows(noma_dir/'fig8_results.csv')
    orr = csv_rows(oma_dir/'fig8_oma'/'fig8_oma_results.csv')
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for branch, color, ls, mk, lab in [
        ('ris',         NOMA_C, '-',  'v', 'NOMA-RIS (MAML-DL)'),
        ('no_ris',      NOMA_C, '--', '^', 'NOMA no-RIS'),
        ('ris_oma',     OMA_C,  '-',  'o', 'OMA-RIS (MAML-DL)'),
        ('no_ris_oma',  OMA_C,  '--', 's', 'OMA no-RIS'),
    ]:
        rows = nr if 'oma' not in branch else orr
        P, R = series(rows, branch)
        if P:
            ax.plot(P, [r*BW_MHZ for r in R], color=color, ls=ls, lw=2.1,
                    marker=mk, ms=8, mfc='white', mec=color, mew=1.6, label=lab)
    ax.set_xlabel('Transmit power budget $P_\\mathrm{max}$ (dBm)', fontsize=12)
    ax.set_ylabel(f'Weighted sum rate (Mbps, BW={BW_MHZ:g} MHz)', fontsize=12)
    ax.set_title('Fig 8 — $R_\\mathrm{sum}$ vs $P_\\mathrm{max}$ : NOMA vs OMA  (N=16, M=2)', fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=10, framealpha=0.92)
    plt.tight_layout(); plt.savefig(out, dpi=220, bbox_inches='tight'); plt.close()
    print(f'saved -> {out}')


def fig4(noma_npz, oma_npz, out):
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for npz, color, lab in [(noma_npz, NOMA_C, 'NOMA'), (oma_npz, OMA_C, 'OMA')]:
        p = Path(npz)
        if not p.exists():
            print(f'[warn] missing {p}'); continue
        d = np.load(p)
        loss = d['iter_losses']
        # light smoothing for readability
        k = max(1, len(loss)//200)
        sm = np.convolve(loss, np.ones(k)/k, mode='valid') if k > 1 else loss
        ax.plot(np.linspace(0, len(loss), len(sm)), sm, color=color, lw=1.8, label=f'{lab} (N=8)')
    ax.set_xlabel('Training iteration', fontsize=12)
    ax.set_ylabel('MAML meta-loss $L$', fontsize=12)
    ax.set_title('Fig 4 — Training convergence (MAML meta-loss)', fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=220, bbox_inches='tight'); plt.close()
    print(f'saved -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=str(Path(__file__).resolve().parent.parent
                                          / 'results_attempt2'))
    args = ap.parse_args()
    base = Path(args.base)
    noma = base/'noma_results'; oma = base/'oma_results'
    outdir = base/'figures'; outdir.mkdir(exist_ok=True)

    fig5(noma, oma, outdir/'fig5_4curve.png')
    fig8(noma, oma, outdir/'fig8_4curve.png')
    fig4(noma/'fig4_iters.npz', oma/'fig4'/'fig4_oma_iters.npz', outdir/'fig4_loss.png')
    print(f'\nFigures in: {outdir}')


if __name__ == '__main__':
    main()

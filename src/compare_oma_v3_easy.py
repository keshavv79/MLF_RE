"""
compare_oma_v3_easy.py
====================================================================
Side-by-side comparison & plot generator for the OMA-TDMA v3-easy policy.
OMA counterpart of compare_v3_easy.py (NOMA).

Key differences from the NOMA evaluator
----------------------------------------
  * The policy output ("alpha") is a TIME split (t_n, t_f, t_T) on the
    simplex, not a power split. Histograms / summary columns are relabelled
    accordingly (t_n / t_f / t_T).
  * OMA physics: per-user MRT, unconstrained sensing BF, NO SIC, no
    inter-user interference. Rates are time-weighted:
        R_k = t_k * log2(1 + g_kk * P_tot / sigma2).

Methods compared (main)
-----------------------
  1. baseline   - random RIS phi + equal time (1/3, 1/3, 1/3)
  2. oma_maml   - MAML bilevel DL (inner-loop phi + AlphaNet time split)

Ablation (decoupling phi vs time-split optimisation)
----------------------------------------------------
  randPhi_randTime  - random phi + random-softmax time
  randPhi_equalTime - random phi + equal time
  randPhi_optTime   - random phi + AlphaNet time (no inner loop)
  randTime_optPhi   - random frozen time + inner-loop optimised phi
  fullOptimized     - oma_maml (inner phi + AlphaNet time)

Outputs (under --out-dir, default ./eval_oma/)
  rate_hist.png, rate_cdf.png, per_leg_rate_cdf.png, alloc_hist.png,
  qos_violations.png, improvement_bars.png, theta_heatmap.png,
  rate_vs_qos_margin.png, summary.csv, summary.txt
  + ablation/ subfolder with the same set.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_dl_v3_easy import load_dataset
from train_dl_oma_maml import (
    AlphaNet, inner_optimize,
    physics_step_oma, rates_from_gains_oma, make_features_oma, lagrangian_loss,
)


# ============================================================
# 1. Helpers
# ============================================================
class _Args:
    """Container for MAML inner-loop args with safe defaults for older ckpts."""

    def __init__(self, d):
        defaults = dict(w_c=0.7, w_s=0.3,
                        inner_steps=5, gamma_theta=0.3,
                        lam_c=2.0, lam_s=50.0, lam_n=15.0, lam_f=20.0,
                        R_th_n=1.0, R_th_f=1.0)
        for k, v in defaults.items():
            setattr(self, k, v)
        for k, v in d.items():
            setattr(self, k, v)


def collect_per_channel(forward_fn, loader, device, R_th_c, R_th_s):
    """Run a forward function on every batch, collect per-channel arrays."""
    Rn, Rf, Rs = [], [], []
    tn, tf, tT = [], [], []
    theta_ang_all = []
    for batch in loader:
        batch = tuple(b.to(device) for b in batch)
        R_n, R_f, R_s, t_alpha, theta = forward_fn(*batch)
        Rn.append(R_n.detach().cpu().numpy())
        Rf.append(R_f.detach().cpu().numpy())
        Rs.append(R_s.detach().cpu().numpy())
        tn.append(t_alpha[:, 0].detach().cpu().numpy())
        tf.append(t_alpha[:, 1].detach().cpu().numpy())
        tT.append(t_alpha[:, 2].detach().cpu().numpy())
        theta_ang_all.append(torch.angle(theta).detach().cpu().numpy())
    Rn = np.concatenate(Rn); Rf = np.concatenate(Rf); Rs = np.concatenate(Rs)
    tn = np.concatenate(tn); tf = np.concatenate(tf); tT = np.concatenate(tT)
    theta_ang = np.concatenate(theta_ang_all, axis=0)
    qc = ((Rn + Rf) < R_th_c).astype(np.float32)
    qs = (Rs < R_th_s).astype(np.float32)
    qany = ((qc + qs) > 0).astype(np.float32)
    Rsum_paper = 0.7 * (Rn + Rf) + 0.3 * Rs
    Rsum_bal   = 0.5 * (Rn + Rf) + 0.5 * Rs
    return dict(Rn=Rn, Rf=Rf, Rs=Rs,
                tn=tn, tf=tf, tT=tT, theta_ang=theta_ang,
                qc=qc, qs=qs, qany=qany,
                Rsum_paper=Rsum_paper, Rsum_bal=Rsum_bal)


# ============================================================
# 2. Per-method forward wrappers
# ============================================================
def _theta_from_phi(phi, ref_dtype):
    return torch.complex(torch.cos(phi), torch.sin(phi)).to(ref_dtype)


def build_baseline_fn(N, cfg, device, seed=0):
    """Random phi + equal time (1/3, 1/3, 1/3)."""
    torch.manual_seed(seed)

    @torch.no_grad()
    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0)
        phi = 2 * np.pi * torch.rand(B, N, device=device)
        gains = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        t_alpha = torch.tensor([1/3, 1/3, 1/3], device=device).expand(B, 3)
        R_n, R_f, R_s = rates_from_gains_oma(gains, t_alpha, cfg['P_tot'], cfg['sigma2'])
        return R_n, R_f, R_s, t_alpha, _theta_from_phi(phi, H_BR.dtype)
    return fw


def build_randPhi_randTime_fn(N, cfg, device, seed=1):
    """Random phi + random-softmax time split."""
    torch.manual_seed(seed)

    @torch.no_grad()
    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0)
        phi = 2 * np.pi * torch.rand(B, N, device=device)
        gains = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        t_alpha = torch.softmax(torch.randn(B, 3, device=device), dim=-1)
        R_n, R_f, R_s = rates_from_gains_oma(gains, t_alpha, cfg['P_tot'], cfg['sigma2'])
        return R_n, R_f, R_s, t_alpha, _theta_from_phi(phi, H_BR.dtype)
    return fw


def build_randPhi_optTime_fn(ckpt_path, cfg, device, seed=2):
    """Random phi + AlphaNet time split (no inner loop)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = AlphaNet(in_dim=6, hidden=ckpt['args'].get('hidden', 256)).to(device)
    net.load_state_dict(ckpt['state_dict']); net.eval()
    torch.manual_seed(seed)

    @torch.no_grad()
    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0)
        phi = 2 * np.pi * torch.rand(B, cfg['N'], device=device)
        gains = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        feats = make_features_oma(gains, cfg['R_th_c'], cfg['R_th_s'],
                                  cfg['P_tot'], cfg['sigma2'])
        t_alpha = net(feats)
        R_n, R_f, R_s = rates_from_gains_oma(gains, t_alpha, cfg['P_tot'], cfg['sigma2'])
        return R_n, R_f, R_s, t_alpha, _theta_from_phi(phi, H_BR.dtype)
    return fw


def build_randTime_optPhi_fn(cfg, device, seed=3, inner_steps=5, gamma_theta=0.3):
    """Random frozen time split + inner-loop optimised phi."""
    torch.manual_seed(seed)

    def fw(H_BR, h_RDn, h_RDf, h_RT, h_TR):
        B = H_BR.size(0)
        dev = H_BR.device
        t_alpha = torch.softmax(torch.randn(B, 3, device=dev), dim=-1).detach()

        phi = 2 * np.pi * torch.rand(B, cfg['N'], device=dev)
        phi.requires_grad_(True)
        with torch.enable_grad():
            for _ in range(inner_steps):
                gains = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
                R_n, R_f, R_s = rates_from_gains_oma(
                    gains, t_alpha, cfg['P_tot'], cfg['sigma2'])
                L, _ = lagrangian_loss(
                    R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
                    R_th_n=0.0, R_th_f=0.0,
                    lam_c=2.0, lam_s=50.0, lam_n=0.0, lam_f=0.0)
                d_phi = torch.autograd.grad(L, phi, create_graph=False)[0]
                phi = phi - gamma_theta * d_phi
            gains = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
            R_n, R_f, R_s = rates_from_gains_oma(
                gains, t_alpha, cfg['P_tot'], cfg['sigma2'])
        return (R_n.detach(), R_f.detach(), R_s.detach(),
                t_alpha, _theta_from_phi(phi.detach(), H_BR.dtype))
    return fw


def build_maml_fn(ckpt_path, cfg, device):
    """Full OMA-MAML: inner-loop phi + AlphaNet time split."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = AlphaNet(in_dim=6, hidden=ckpt['args'].get('hidden', 256)).to(device)
    net.load_state_dict(ckpt['state_dict']); net.eval()
    inner_args = _Args(ckpt['args'])

    def fw(*batch):
        with torch.enable_grad():
            _, _, R_n, R_f, R_s, phi, t_alpha = inner_optimize(
                net, batch, cfg, inner_args, create_meta_graph=False)
        return (R_n.detach(), R_f.detach(), R_s.detach(),
                t_alpha.detach(), _theta_from_phi(phi.detach(), batch[0].dtype))
    return fw


# ============================================================
# 3. Plots
# ============================================================
COLORS = {
    'baseline':          '#888888',
    'oma_maml':          '#1f77b4',
    # ablation
    'randPhi_randTime':  '#cccccc',
    'randPhi_equalTime': '#bcbd22',
    'randPhi_optTime':   '#ff7f0e',
    'randTime_optPhi':   '#17becf',
    'fullOptimized':     '#d62728',
}


def plot_rate_hist(results, out_path, key='Rsum_paper', title_suffix='(paper 0.7/0.3)'):
    plt.figure(figsize=(9, 5.5))
    for name, r in results.items():
        plt.hist(r[key], bins=50, alpha=0.55, label=name,
                 color=COLORS.get(name), edgecolor='none')
    plt.xlabel('R_DL_sum (bits/s/Hz)'); plt.ylabel('Count')
    plt.title(f'OMA weighted sum-rate distribution {title_suffix}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def plot_rate_cdf(results, out_path, key='Rsum_paper', title_suffix='(paper 0.7/0.3)'):
    plt.figure(figsize=(9, 5.5))
    for name, r in results.items():
        xs = np.sort(r[key]); ys = np.linspace(0, 1, xs.size)
        plt.plot(xs, ys, label=name, color=COLORS.get(name), lw=2)
    plt.xlabel('R_DL_sum (bits/s/Hz)'); plt.ylabel('CDF')
    plt.title(f'OMA weighted sum-rate CDF {title_suffix}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def plot_per_leg_rate_cdf(results, out_path, R_th_c, R_th_s):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, key, label, thresh in [
        (axes[0], 'Rn', 'R_n  (near user)', None),
        (axes[1], 'Rf', 'R_f  (far user)',  None),
        (axes[2], 'Rs', 'R_s  (sensing)',   R_th_s),
    ]:
        for name, r in results.items():
            xs = np.sort(r[key]); ys = np.linspace(0, 1, xs.size)
            ax.plot(xs, ys, label=name, color=COLORS.get(name), lw=2)
        if thresh is not None:
            ax.axvline(thresh, color='black', linestyle=':', alpha=0.6,
                       label=f'R_th = {thresh}')
        ax.set_xlabel('Rate (bits/s/Hz)'); ax.set_title(label)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel('CDF')
    fig.suptitle(f'OMA per-leg rate CDFs  (R_th_c={R_th_c} on R_n+R_f, '
                 f'R_th_s={R_th_s} on R_s)', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_path, dpi=200); plt.close()


def plot_alloc_hist(results, out_path):
    """Time-split distributions (t_n / t_f / t_T)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, label in [
        (axes[0], 'tn', 't_n  (near-user time share)'),
        (axes[1], 'tf', 't_f  (far-user time share)'),
        (axes[2], 'tT', 't_T  (sensing time share)'),
    ]:
        for name, r in results.items():
            ax.hist(r[key], bins=40, alpha=0.55, label=name,
                    color=COLORS.get(name), edgecolor='none')
        ax.set_xlabel('Fraction of frame time'); ax.set_title(label)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel('Count')
    fig.suptitle('Learned OMA time-split distributions', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_path, dpi=200); plt.close()


def plot_qos_violations(results, out_path):
    labels = list(results.keys())
    qany = [results[k]['qany'].mean() for k in labels]
    qc   = [results[k]['qc'].mean() for k in labels]
    qs   = [results[k]['qs'].mean() for k in labels]
    x = np.arange(len(labels)); width = 0.27
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width, qany, width, label='qos_viol_any', color='#444444')
    plt.bar(x,         qc,   width, label='qc  (R_n+R_f < R_th_c)', color='#1f77b4')
    plt.bar(x + width, qs,   width, label='qs  (R_s   < R_th_s)',   color='#d62728')
    plt.xticks(x, labels, rotation=15); plt.ylabel('Violation fraction')
    plt.title('OMA QoS constraint violation rates (lower is better)')
    plt.legend(); plt.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(qany):
        plt.text(i - width, v + 0.005, f'{100*v:.1f}%', ha='center', fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def plot_improvement_bars(results, baseline_name, out_path):
    labels = [k for k in results.keys() if k != baseline_name]
    metrics = [('Rsum_paper', 'R_DL_sum', False),
               ('Rn', 'R_n', False), ('Rf', 'R_f', False), ('Rs', 'R_s', False),
               ('qany', 'qos_viol_any', True), ('qs', 'qs', True), ('qc', 'qc', True)]
    base_means = {m: results[baseline_name][m].mean() for m, _, _ in metrics}
    n_methods = len(labels); n_metrics = len(metrics)
    width = 0.8 / max(n_methods, 1); x = np.arange(n_metrics)
    plt.figure(figsize=(12, 6))
    for i, name in enumerate(labels):
        deltas = []
        for m, _, lower in metrics:
            b = base_means[m]; v = results[name][m].mean()
            if abs(b) < 1e-9:
                deltas.append(0.0); continue
            pct = 100 * (v - b) / abs(b)
            if lower: pct = -pct
            deltas.append(max(min(pct, 500), -500))
        plt.bar(x + (i - (n_methods - 1)/2) * width, deltas, width,
                label=name, color=COLORS.get(name))
    plt.axhline(0, color='black', lw=0.7)
    plt.xticks(x, [lab for _, lab, _ in metrics], rotation=15)
    plt.ylabel('% improvement vs baseline  (capped at ±500)')
    plt.title('OMA per-metric improvement vs baseline (higher = better)')
    plt.legend(); plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def plot_rate_vs_qos_margin(results, out_path, R_th_c, R_th_s,
                            key='Rsum_paper', title_suffix='(paper 0.7/0.3)'):
    plt.figure(figsize=(10, 6))
    for name, r in results.items():
        margin = np.minimum((r['Rn'] + r['Rf']) - R_th_c, r['Rs'] - R_th_s)
        plt.scatter(margin, r[key], s=6, alpha=0.35,
                    label=name, color=COLORS.get(name))
    plt.axvline(0, color='black', lw=0.7, linestyle='--', alpha=0.7)
    plt.xlabel('Min QoS margin  =  min( R_n+R_f - R_th_c ,  R_s - R_th_s )  [bits/s/Hz]')
    plt.ylabel('R_DL_sum (bits/s/Hz)')
    plt.title(f'OMA rate vs QoS margin {title_suffix}  (top-right = feasible & high rate)')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def plot_theta_heatmap(results, out_path):
    names = list(results.keys()); n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.0 + 0.8 * n), squeeze=False)
    for i, name in enumerate(names):
        ang = results[name]['theta_ang']                      # (B, N)
        mean_ang = np.angle(np.exp(1j * ang).mean(axis=0))    # circular mean
        ax = axes[i, 0]
        ax.imshow(mean_ang[np.newaxis, :], aspect='auto', cmap='twilight',
                  vmin=-np.pi, vmax=np.pi)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=9)
        if i == n - 1:
            ax.set_xlabel('RIS element index')
            ax.set_xticks(np.arange(mean_ang.size))
        else:
            ax.set_xticks([])
    fig.suptitle('Mean RIS phase per element (circular mean across test channels)',
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_path, dpi=200); plt.close()


# ============================================================
# 4. Summary writers
# ============================================================
def write_summary(results, baseline_name, R_th_c, R_th_s, out_dir):
    csv_path = out_dir / 'summary.csv'
    txt_path = out_dir / 'summary.txt'
    header = ['method', 'R_DL_sum_paper', 'R_DL_sum_balanced',
              'R_n', 'R_f', 'R_s', 'qos_viol_any', 'qc', 'qs',
              't_n', 't_f', 't_T']
    with csv_path.open('w') as f:
        f.write(','.join(header) + '\n')
        for name, r in results.items():
            row = [name,
                   f"{r['Rsum_paper'].mean():.4f}", f"{r['Rsum_bal'].mean():.4f}",
                   f"{r['Rn'].mean():.4f}", f"{r['Rf'].mean():.4f}", f"{r['Rs'].mean():.4f}",
                   f"{r['qany'].mean():.4f}", f"{r['qc'].mean():.4f}", f"{r['qs'].mean():.4f}",
                   f"{r['tn'].mean():.4f}", f"{r['tf'].mean():.4f}", f"{r['tT'].mean():.4f}"]
            f.write(','.join(row) + '\n')

    base = results[baseline_name]
    with txt_path.open('w') as f:
        f.write("OMA v3-easy test-set comparison\n")
        f.write(f"Samples : {next(iter(results.values()))['Rn'].size}\n")
        f.write(f"R_th_c  : {R_th_c}  (R_n + R_f >= R_th_c)\n")
        f.write(f"R_th_s  : {R_th_s}  (R_s >= R_th_s)\n\n")
        f.write(f"{'Method':<18}{'Rsum(7/3)':>12}{'Rsum(5/5)':>12}"
                f"{'R_n':>8}{'R_f':>8}{'R_s':>8}{'qany':>9}{'qc':>9}{'qs':>9}\n")
        f.write('-' * 91 + '\n')
        for name, r in results.items():
            f.write(f"{name:<18}{r['Rsum_paper'].mean():>12.4f}{r['Rsum_bal'].mean():>12.4f}"
                    f"{r['Rn'].mean():>8.3f}{r['Rf'].mean():>8.3f}{r['Rs'].mean():>8.3f}"
                    f"{r['qany'].mean():>9.4f}{r['qc'].mean():>9.4f}{r['qs'].mean():>9.4f}\n")
        f.write(f"\n% improvement vs {baseline_name}:\n")
        for name, r in results.items():
            if name == baseline_name: continue
            f.write(f"  {name}:\n")
            for k, lab, lower in [('Rsum_paper', 'R_DL_sum (7/3)', False),
                                  ('Rsum_bal',   'R_DL_sum (5/5)', False),
                                  ('Rn', 'R_n', False), ('Rf', 'R_f', False),
                                  ('Rs', 'R_s', False), ('qany', 'qos_viol_any', True),
                                  ('qc', 'qc', True), ('qs', 'qs', True)]:
                b = base[k].mean(); v = r[k].mean()
                pct = 100 * (v - b) / abs(b + 1e-12)
                if lower: pct = -pct
                f.write(f"    {lab:<20} {pct:+.1f}%\n")
    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved summary TXT: {txt_path}")


def render_all(results, out_dir, R_th_c, R_th_s, baseline_name, suffix):
    plot_rate_hist(results, out_dir / 'rate_hist.png', 'Rsum_paper', suffix)
    plot_rate_cdf(results, out_dir / 'rate_cdf.png', 'Rsum_paper', suffix)
    plot_per_leg_rate_cdf(results, out_dir / 'per_leg_rate_cdf.png', R_th_c, R_th_s)
    plot_alloc_hist(results, out_dir / 'alloc_hist.png')
    plot_qos_violations(results, out_dir / 'qos_violations.png')
    plot_improvement_bars(results, baseline_name, out_dir / 'improvement_bars.png')
    plot_theta_heatmap(results, out_dir / 'theta_heatmap.png')
    plot_rate_vs_qos_margin(results, out_dir / 'rate_vs_qos_margin.png',
                            R_th_c, R_th_s, 'Rsum_paper', suffix)
    write_summary(results, baseline_name, R_th_c, R_th_s, out_dir)


# ============================================================
# 5. Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mat',     default='ISAC_RIS_OMA_channels_v3_easy_TEST.mat')
    p.add_argument('--out-dir', default='eval_oma')
    p.add_argument('--batch',   type=int, default=256)
    p.add_argument('--device',  default=('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--ckpt_maml', default='policy_oma_maml_best.pt')
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ds = load_dataset(Path(args.mat))
    N, M = ds['N'], ds['M']
    cfg = dict(N=N, M=M, P_tot=ds['P_tot'], sigma2=ds['sigma2'], beta_T=ds['beta_T'],
               R_th_c=ds['R_th_c'], R_th_s=ds['R_th_s'])
    R_th_c, R_th_s = ds['R_th_c'], ds['R_th_s']

    tensors = tuple(torch.from_numpy(ds[k])
                    for k in ['H_BR', 'h_RDn', 'h_RDf', 'h_RT', 'h_TR'])
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch)
    n_samples = tensors[0].shape[0]
    print(f"OMA test set: {args.mat}  ({n_samples} channels, N={N}, M={M})")
    print(f"R_th_c={R_th_c}  R_th_s={R_th_s}\n")

    # ---- Main methods ----
    results = {}
    print('Running baseline (random RIS + equal time 1/3,1/3,1/3) ...')
    results['baseline'] = collect_per_channel(
        build_baseline_fn(N, cfg, device), loader, device, R_th_c, R_th_s)

    if os.path.exists(args.ckpt_maml):
        print(f'Running OMA-MAML from {args.ckpt_maml} ...')
        results['oma_maml'] = collect_per_channel(
            build_maml_fn(args.ckpt_maml, cfg, device), loader, device, R_th_c, R_th_s)
    else:
        print(f'[warn] checkpoint not found: {args.ckpt_maml} (skipping MAML method)')

    print('\nGenerating main plots ...')
    render_all(results, out_dir, R_th_c, R_th_s, 'baseline', '(paper 0.7/0.3)')

    # ---- Ablation ----
    if os.path.exists(args.ckpt_maml):
        print('\nRunning ablation cases ...')
        ab = {}
        ab['randPhi_randTime']  = collect_per_channel(
            build_randPhi_randTime_fn(N, cfg, device), loader, device, R_th_c, R_th_s)
        ab['randPhi_equalTime'] = results['baseline']
        ab['randPhi_optTime']   = collect_per_channel(
            build_randPhi_optTime_fn(args.ckpt_maml, cfg, device),
            loader, device, R_th_c, R_th_s)
        ab['randTime_optPhi']   = collect_per_channel(
            build_randTime_optPhi_fn(cfg, device), loader, device, R_th_c, R_th_s)
        ab['fullOptimized']     = results['oma_maml']

        ab_dir = out_dir / 'ablation'; ab_dir.mkdir(exist_ok=True)
        print('Generating ablation plots ...')
        render_all(ab, ab_dir, R_th_c, R_th_s, 'randPhi_randTime',
                   '(ablation, paper 0.7/0.3)')

        print('\n' + '=' * 80)
        print('OMA ABLATION RESULTS  (decoupling phi vs time-split optimisation)')
        print('=' * 80)
        print(f"{'Case':<20}{'Rsum(7/3)':>12}{'R_n':>8}{'R_f':>8}{'R_s':>8}{'qany':>10}")
        print('-' * 70)
        for name, r in ab.items():
            print(f"{name:<20}{r['Rsum_paper'].mean():>12.4f}{r['Rn'].mean():>8.3f}"
                  f"{r['Rf'].mean():>8.3f}{r['Rs'].mean():>8.3f}{r['qany'].mean():>10.4f}")
        print(f"Ablation outputs -> {ab_dir.resolve()}")

    # ---- Headline ----
    print('\n' + '=' * 80)
    print('OMA HEADLINE RESULTS')
    print('=' * 80)
    print(f"{'Method':<18}{'Rsum(7/3)':>12}{'Rsum(5/5)':>12}"
          f"{'R_n':>8}{'R_f':>8}{'R_s':>8}{'qany':>10}{'qc':>9}{'qs':>9}")
    print('-' * 92)
    for name, r in results.items():
        print(f"{name:<18}{r['Rsum_paper'].mean():>12.4f}{r['Rsum_bal'].mean():>12.4f}"
              f"{r['Rn'].mean():>8.3f}{r['Rf'].mean():>8.3f}{r['Rs'].mean():>8.3f}"
              f"{r['qany'].mean():>10.4f}{r['qc'].mean():>9.4f}{r['qs'].mean():>9.4f}")
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == '__main__':
    main()

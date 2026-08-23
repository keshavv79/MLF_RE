"""
train_dl_v3_easy_maml.py
====================================================================
MAML-style bilevel training for joint optimisation of RIS phase shifts
and NOMA power allocation on the MISO RIS-ISAC-NOMA v3-easy dataset.

This is the bilevel counterpart to train_dl_v3_easy.py.  It mirrors
the algorithmic structure of training.py (paper-style MAML) but uses
the correct full MISO physics from train_dl_v3_easy.py:
    - 4-hop sensing channel G_T (M x M)
    - Cluster MRT comm beamformer + null-space sensing beamformer
    - Sum-rate QoS (R_n + R_f >= R_th_c), not per-user

Algorithmic structure
---------------------
    Theta is NOT predicted by a network.  It is a free per-channel
    variable phi in R^N, where theta_i = exp(j*phi_i).

    AlphaNet predicts alpha = (a_n, a_f, a_T) on the simplex, given
    features that depend on the *current* phi (the effective scalar
    gains after cascaded MISO + beamforming).

    Inner loop (K steps, per batch):
        for k in 1..K:
            theta = exp(j*phi)
            gains = MISO_physics(theta, channels)
            alpha = AlphaNet(features(gains, qos))
            (R_n, R_f, R_s) = SINRs(gains, alpha)
            L = -E[R_sum] + Lagrangian_pen
            phi  <-  phi - gamma * d L / d phi
                     (create_graph=True so meta-grad flows)

    Outer step:
        final_loss.backward()   # meta-grad through K inner steps
        optimizer.step()        # update AlphaNet parameters

Inference
---------
    Same procedure: run the K-step inner loop on each new channel to
    obtain (theta*, alpha*).  Slower than single-pass DL but closer
    to AO quality per channel.

Output: policy_maml_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from train_dl_v3_easy import load_dataset


# ============================================================
# 1. AlphaNet: predicts (a_n, a_f, a_T) on the simplex
# ============================================================
class AlphaNet(nn.Module):
    """MLP with LayerNorm + LeakyReLU; outputs softmax over 3 power slots."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


# ============================================================
# 2. MISO physics (alpha-independent up to "gains")
# ============================================================
def physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T):
    """
    Cascaded MISO physics + closed-form beamformers, returning the six
    alpha-independent effective scalar gains used in the NOMA SINRs.
    """
    theta = torch.complex(torch.cos(phi), torch.sin(phi)).to(H_BR.dtype)
    B, N, M = H_BR.shape
    cdtype = H_BR.dtype

    TH = theta.unsqueeze(-1) * H_BR
    h_n = torch.einsum("bn,bnm->bm", h_RDn, TH)
    h_f = torch.einsum("bn,bnm->bm", h_RDf, TH)

    th_tr = theta * h_TR
    v1 = torch.einsum("bnm,bn->bm", H_BR.conj(), th_tr)
    v2 = torch.einsum("bn,bnm->bm", h_RT, TH)
    G_T = (beta_T ** 0.5) * v1.unsqueeze(-1) * v2.unsqueeze(-2)

    # FAIR weighted common beamformer (w_n = w_f = 0.7 h_f + 0.3 h_n)
    w_c = 0.7 * h_f + 0.3 * h_n
    w_c = w_c / (w_c.norm(dim=-1, keepdim=True) + 1e-9)

    # FAIR soft sensing nulling: A = G^H G - lambda * H_u H_u^H  (lambda=0.5).
    # Closed-form sensing BF = top eigenvector of A. The complex eigenvector is
    # phase-ambiguous, so eigh-backward can be ill-defined on modern torch.
    # Treat w_T as a fixed closed-form direction (no grad through eigh); the
    # sensing rate still depends on phi through G_T in gw/gsT below. Mirrors
    # miso_isac_noma_v3_fair_chatpgt.m.
    H_u = torch.stack([h_n, h_f], dim=-1)              # (B, M, 2)
    HuHu = H_u @ H_u.conj().transpose(-1, -2)          # (B, M, M)
    with torch.no_grad():
        A = G_T.conj().transpose(-1, -2) @ G_T - 0.5 * HuHu
        A = 0.5 * (A + A.conj().transpose(-1, -2))
        # Use general eig (matches MATLAB; robust to the near-singular,
        # repeated-eigenvalue matrices where eigh fails to converge). Normalise
        # A to O(1) first (tiny channel magnitudes). w_T = eigvec of max real eig.
        scale = torch.amax(A.abs(), dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        A = A / scale
        evals, evecs = torch.linalg.eig(A)
        top = evals.real.argmax(dim=-1)
        w_T = evecs.gather(-1, top[:, None, None].expand(-1, A.shape[-1], 1)).squeeze(-1)
        w_T = w_T / (w_T.norm(dim=-1, keepdim=True) + 1e-9)
    w_T = w_T.detach().to(G_T.dtype)

    gw = torch.einsum("bmn,bn->bm", G_T, w_T)
    u_T = gw / (gw.norm(dim=-1, keepdim=True) + 1e-9)

    def absq(a, b):
        return (a.conj() * b).sum(-1).abs() ** 2

    def gscat(u, G, w):
        Gw = torch.einsum("bmn,bn->bm", G, w)
        return (u.conj() * Gw).sum(-1).abs() ** 2

    gff = absq(h_f, w_c)
    gfn = absq(h_f, w_c)   # w_n = w_c
    gfT = absq(h_f, w_T)
    gnn = absq(h_n, w_c)
    gnT = absq(h_n, w_T)
    gsT = gscat(u_T, G_T, w_T)
    return dict(gff=gff, gfn=gfn, gfT=gfT, gnn=gnn, gnT=gnT, gsT=gsT)


def rates_from_gains(gains, alpha, P_tot, sigma2):
    a_n, a_f, a_T = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    Pn = a_n * P_tot; Pf = a_f * P_tot; PT = a_T * P_tot
    snr_f = (gains['gff'] * Pf) / (gains['gfn'] * Pn + gains['gfT'] * PT + sigma2)
    snr_n = (gains['gnn'] * Pn) / (gains['gnT'] * PT + sigma2)
    snr_s = (gains['gsT'] * PT) / sigma2
    R_n = torch.log2(1.0 + snr_n)
    R_f = torch.log2(1.0 + snr_f)
    R_s = torch.log2(1.0 + snr_s)
    return R_n, R_f, R_s


def make_features(gains, R_th_c, R_th_s, P_tot, sigma2):
    """6 gain-features in dB + 2 QoS + 1 SNR-budget -> 9-dim input."""
    eps = 1e-20
    g_db = [10.0 * torch.log10(gains[k] + eps)
            for k in ('gnn', 'gff', 'gsT', 'gfn', 'gfT', 'gnT')]
    feats = torch.stack(g_db, dim=-1).float()
    B = feats.shape[0]
    qos = torch.tensor([R_th_c, R_th_s], dtype=torch.float32,
                       device=feats.device).expand(B, 2)
    snr_db = torch.full((B, 1), float(10.0 * np.log10(P_tot / sigma2)),
                        dtype=torch.float32, device=feats.device)
    return torch.cat([feats, qos, snr_db], dim=-1)


# ============================================================
# 3. Lagrangian loss (P3 + optional NOMA-fairness floors)
# ============================================================
def lagrangian_loss(R_n, R_f, R_s, R_th_c, R_th_s,
                    R_th_n=0.0, R_th_f=0.0,
                    w_c=0.7, w_s=0.3,
                    lam_c=2.0, lam_s=50.0,
                    lam_n=0.0, lam_f=0.0):
    R_sum = w_c * (R_n + R_f) + w_s * R_s
    pen_c = torch.relu(R_th_c - (R_n + R_f))
    pen_s = torch.relu(R_th_s - R_s)
    pen_n = torch.relu(R_th_n - R_n)
    pen_f = torch.relu(R_th_f - R_f)
    L = (-R_sum.mean()
         + lam_c * pen_c.mean() + lam_s * pen_s.mean()
         + lam_n * pen_n.mean() + lam_f * pen_f.mean())
    return L, R_sum


# ============================================================
# 4. Inner loop: per-channel phi optimisation, network in the middle
# ============================================================
def inner_optimize(model, batch, cfg, args, create_meta_graph: bool):
    """
    K-step gradient descent on phi.  Each inner step:
      - compute physics & gains from current phi
      - query AlphaNet for current alpha
      - compute L and grad_phi L (with create_graph=create_meta_graph)
      - phi  <-  phi - gamma * grad_phi

    Returns (final_L, R_sum_mean, R_n, R_f, R_s, phi_final, alpha_final).
    During training pass create_meta_graph=True so .backward() on final_L
    can flow through all K steps into AlphaNet weights.
    """
    H_BR, h_RDn, h_RDf, h_RT, h_TR = batch
    B, N, _ = H_BR.shape
    device = H_BR.device

    phi = (2.0 * np.pi) * torch.rand(B, N, device=device)
    phi.requires_grad_(True)

    def _inner_loss(phi):
        gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        feats = make_features(gains, cfg['R_th_c'], cfg['R_th_s'],
                              cfg['P_tot'], cfg['sigma2'])
        alpha = model(feats)
        R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
        L, _ = lagrangian_loss(
            R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
            R_th_n=args.R_th_n, R_th_f=args.R_th_f,
            lam_c=args.lam_c, lam_s=args.lam_s,
            lam_n=args.lam_n, lam_f=args.lam_f,
            w_c=args.w_c, w_s=args.w_s)
        return L

    # Inner loop: solve phi. For FOMAML (create_meta_graph=False) solve it HARD
    # with Adam so the FULL achievable RIS gain is extracted -- 5-step plain GD
    # under-solves at high SINR where dR/dphi ~ 1/(1+SINR) is vanishingly small.
    # The second-order path keeps functional GD (unused; kept for completeness).
    if create_meta_graph:
        for _ in range(args.inner_steps):
            d_phi = torch.autograd.grad(_inner_loss(phi), phi, create_graph=True)[0]
            phi = phi - args.gamma_theta * d_phi
    else:
        inner_opt = torch.optim.Adam([phi], lr=getattr(args, 'inner_lr', 0.1))
        for _ in range(args.inner_steps):
            phi.grad = torch.autograd.grad(_inner_loss(phi), phi)[0]
            inner_opt.step()
        phi = phi.detach()

    # Final pass at the converged phi
    gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
    feats = make_features(gains, cfg['R_th_c'], cfg['R_th_s'],
                          cfg['P_tot'], cfg['sigma2'])
    alpha = model(feats)
    R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
    L_final, R_sum = lagrangian_loss(
        R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
        R_th_n=args.R_th_n, R_th_f=args.R_th_f,
        lam_c=args.lam_c, lam_s=args.lam_s,
        lam_n=args.lam_n, lam_f=args.lam_f,
    )
    return L_final, R_sum.mean(), R_n, R_f, R_s, phi, alpha


# ============================================================
# 5. Baseline for sanity reference (random phi, fixed alloc)
# ============================================================
@torch.no_grad()
def baseline_eval(loader, device, cfg, a_baseline=(0.30, 0.40, 0.30)):
    N = cfg['N']
    P_tot, sigma2, beta_T = cfg['P_tot'], cfg['sigma2'], cfg['beta_T']
    R_th_c, R_th_s = cfg['R_th_c'], cfg['R_th_s']
    R_sum_tot, qany_tot, n_tot = 0.0, 0.0, 0
    for batch in loader:
        H_BR, h_RDn, h_RDf, h_RT, h_TR = [b.to(device) for b in batch]
        B = H_BR.shape[0]
        phi = 2.0 * np.pi * torch.rand(B, N, device=device)
        gains = physics_step(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T)
        alpha = torch.tensor(a_baseline, dtype=torch.float32,
                             device=device).expand(B, 3)
        R_n, R_f, R_s = rates_from_gains(gains, alpha, P_tot, sigma2)
        R_sum = 0.7 * (R_n + R_f) + 0.3 * R_s
        qany = (((R_n + R_f) < R_th_c) | (R_s < R_th_s)).float()
        R_sum_tot += R_sum.sum().item()
        qany_tot += qany.sum().item()
        n_tot += B
    return R_sum_tot / n_tot, qany_tot / n_tot


# ============================================================
# 6. Train / eval driver
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat",         default="ISAC_RIS_NOMA_channels_v3_fair.mat")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch",       type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--inner_steps", type=int,   default=10,
                        help="inner-loop steps to solve RIS phi (Adam, FOMAML)")
    parser.add_argument("--inner_lr",    type=float, default=0.1,
                        help="Adam lr for the inner RIS-phi optimisation")
    parser.add_argument("--gamma_theta", type=float, default=0.3,
                        help="step size for the (unused) 2nd-order GD inner path")
    parser.add_argument("--lam_c",       type=float, default=2.0)
    parser.add_argument("--lam_s",       type=float, default=50.0)
    parser.add_argument("--lam_n",       type=float, default=15.0)
    parser.add_argument("--lam_f",       type=float, default=20.0)
    parser.add_argument("--R_th_n",      type=float, default=1.0)
    parser.add_argument("--R_th_f",      type=float, default=1.0)
    parser.add_argument("--w_c",         type=float, default=0.7,
                        help="weight on comm sum-rate in P3 objective (eq. 23)")
    parser.add_argument("--w_s",         type=float, default=0.3,
                        help="weight on sensing rate in P3 objective (eq. 23)")
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--seed",        type=int,   default=7)
    parser.add_argument("--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out",         default="policy_noma_fair_best.pt")
    parser.add_argument("--iter_log",    default=None,
                        help="Path to save per-iteration training losses (.npz). "
                             "Defaults to <out>_iters.npz.")
    parser.add_argument("--P_tot_dBm",   type=float, default=None,
                        help="Override P_tot (dBm). If unset, uses value from .mat. "
                             "Channels are P-independent so overriding is sound.")
    args = parser.parse_args()
    if args.iter_log is None:
        args.iter_log = str(Path(args.out).with_suffix("")) + "_iters.npz"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    # ---- data ----
    ds = load_dataset(Path(args.mat))
    P_tot_eff = ds['P_tot']
    if args.P_tot_dBm is not None:
        P_tot_eff = 10.0 ** ((args.P_tot_dBm - 30.0) / 10.0)
        print(f"[override] P_tot {ds['P_tot']:.3e} -> {P_tot_eff:.3e}  "
              f"({args.P_tot_dBm:.1f} dBm)")
    cfg = dict(
        N=ds['N'], M=ds['M'],
        P_tot=P_tot_eff, sigma2=ds['sigma2'], beta_T=ds['beta_T'],
        R_th_c=ds['R_th_c'], R_th_s=ds['R_th_s'],
    )
    tensors = (torch.from_numpy(ds['H_BR']),
               torch.from_numpy(ds['h_RDn']),
               torch.from_numpy(ds['h_RDf']),
               torch.from_numpy(ds['h_RT']),
               torch.from_numpy(ds['h_TR']))
    full = TensorDataset(*tensors)
    n_train = int(0.9 * len(full))
    n_val = len(full) - n_train
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    print(f"Samples train={n_train} val={n_val} | N={cfg['N']} M={cfg['M']}")
    print(f"R_th_c={cfg['R_th_c']}  R_th_s={cfg['R_th_s']}  P_tot={cfg['P_tot']:.3e}")
    print(f"MAML: inner_steps={args.inner_steps}  gamma_theta={args.gamma_theta}  outer_lr={args.lr}")
    print(f"lam (c/s/n/f) = {args.lam_c}/{args.lam_s}/{args.lam_n}/{args.lam_f}")

    R_base, q_base = baseline_eval(val_loader, device, cfg, a_baseline=(0.20, 0.55, 0.25))
    print(f"[Baseline] random-phi + (0.30/0.40/0.30):  "
          f"R_DL_sum={R_base:.4f}  qos_viol_any={q_base:.4f}\n")

    # ---- model ----
    model = AlphaNet(in_dim=9, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iter_losses = []
    iter_R = []

    def epoch_pass(loader, train_mode):
        model.train(train_mode)
        agg = dict(L=0, R=0, qany=0, qc=0, qs=0, n=0)
        for batch in loader:
            batch = tuple(b.to(device) for b in batch)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                # First-order MAML (FOMAML): the M=4 fair model has huge SINR
                # (R_n~21), so the second-order meta-gradient is non-finite
                # (gnorm=inf). First-order is stable and standard.
                L, R_sum, R_n, R_f, R_s, _, _ = inner_optimize(
                    model, batch, cfg, args, create_meta_graph=False)
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                iter_losses.append(L.item())
                iter_R.append(R_sum.item())
            else:
                with torch.enable_grad():
                    L, R_sum, R_n, R_f, R_s, _, _ = inner_optimize(
                        model, batch, cfg, args, create_meta_graph=False)
            B = batch[0].size(0)
            qc = ((R_n + R_f) < cfg['R_th_c']).float()
            qs = (R_s < cfg['R_th_s']).float()
            qany = ((qc + qs) > 0).float()
            agg['L']    += L.item() * B
            agg['R']    += R_sum.item() * B
            agg['qc']   += qc.mean().item() * B
            agg['qs']   += qs.mean().item() * B
            agg['qany'] += qany.mean().item() * B
            agg['n']    += B
        return {k: v / agg['n'] for k, v in agg.items() if k != 'n'}

    print(f"{'epoch':>5} | {'L_tr':>7} {'R_tr':>6} {'qany_tr':>7} | "
          f"{'L_val':>7} {'R_val':>6} {'qc_v':>5} {'qs_v':>5} {'qany_v':>7}")

    best_score = -float("inf")
    for ep in range(1, args.epochs + 1):
        tr = epoch_pass(train_loader, True)
        va = epoch_pass(val_loader, False)
        print(f"{ep:5d} | {tr['L']:7.3f} {tr['R']:6.3f} {tr['qany']:7.3f} | "
              f"{va['L']:7.3f} {va['R']:6.3f} "
              f"{va['qc']:5.3f} {va['qs']:5.3f} {va['qany']:7.3f}")
        score = va['R'] - 5.0 * va['qany']
        if score > best_score:
            best_score = score
            torch.save({
                'state_dict': model.state_dict(),
                'args': vars(args),
                'cfg': cfg,
                'val': va,
                'baseline': {'R': R_base, 'qany': q_base},
            }, args.out)

    print(f"\nBest score (R - 5*qany) = {best_score:.4f}  ->  {args.out}")
    print(f"Baseline reference         : R={R_base:.4f}  qany={q_base:.4f}")

    np.savez(args.iter_log,
             iter_losses=np.array(iter_losses, dtype=np.float32),
             iter_R=np.array(iter_R, dtype=np.float32),
             batch=args.batch, lr=args.lr,
             epochs=args.epochs, inner_steps=args.inner_steps)
    print(f"Per-iter log saved: {args.iter_log}  ({len(iter_losses)} steps)")


if __name__ == "__main__":
    main()

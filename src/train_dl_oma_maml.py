"""
train_dl_oma_maml.py
====================================================================
MAML bilevel training for OMA-TDMA on the MISO RIS-ISAC v3-easy dataset.

OMA system model — 3 orthogonal time slots, shared RIS phase vector phi:
  Slot 1 (near user) : w_n = h_n / ||h_n||   per-user MRT,  full P_tot
  Slot 2 (far user)  : w_f = h_f / ||h_f||   per-user MRT,  full P_tot
  Slot 3 (sensing)   : w_T = dom. eigvec of G_T^H G_T,      full P_tot

  Gains (no cross-interference):
    g_nn = ||h_n||^2          g_ff = ||h_f||^2          g_sT = ||G_T w_T||^2

  Rates:
    R_n = t_n * log2(1 + g_nn * P_tot / sigma2)
    R_f = t_f * log2(1 + g_ff * P_tot / sigma2)
    R_s = t_T * log2(1 + g_sT * P_tot / sigma2)

  AlphaNet predicts (t_n, t_f, t_T) on the simplex (in_dim=6).
  Inner loop optimises phi (shared RIS phase shifts).
  Outer loop updates AlphaNet via MAML meta-gradient.

Output: policy_oma_maml_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from train_dl_v3_easy import load_dataset


# ============================================================
# 1. AlphaNet — predicts (t_n, t_f, t_T) on the simplex
#    in_dim=6: 3 gains (dB) + 2 QoS thresholds + 1 SNR budget
# ============================================================
class AlphaNet(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


# ============================================================
# 2. OMA physics — per-user MRT, unconstrained sensing BF
# ============================================================
def physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T):
    """
    Cascaded MISO physics for OMA-TDMA. Single shared phi across all slots.
    Returns g_nn, g_ff, g_sT — three interference-free gains.
    """
    theta = torch.complex(torch.cos(phi), torch.sin(phi)).to(H_BR.dtype)
    B, N, M = H_BR.shape

    TH  = theta.unsqueeze(-1) * H_BR
    h_n = torch.einsum("bn,bnm->bm", h_RDn, TH)   # (B, M) cascaded near-user channel
    h_f = torch.einsum("bn,bnm->bm", h_RDf, TH)   # (B, M) cascaded far-user channel

    th_tr = theta * h_TR
    v1    = torch.einsum("bnm,bn->bm", H_BR.conj(), th_tr)
    v2    = torch.einsum("bn,bnm->bm", h_RT, TH)
    G_T   = (beta_T ** 0.5) * v1.unsqueeze(-1) * v2.unsqueeze(-2)  # (B, M, M)

    # Slot 1 & 2: per-user MRT (separate slots, no interference)
    w_n = h_n / (h_n.norm(dim=-1, keepdim=True) + 1e-9)
    w_f = h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-9)

    # Slot 3: dominant eigenvector of G_T^H G_T. Closed-form, no grad through
    # the phase-ambiguous eigh; tiny relative jitter so eigh converges on the
    # low-rank (rank-1 -> repeated-eigenvalue) sensing matrix at M>=4.
    with torch.no_grad():
        GHG = G_T.conj().transpose(-1, -2) @ G_T
        GHG = 0.5 * (GHG + GHG.conj().transpose(-1, -2))  # enforce Hermitian
        scale = torch.amax(GHG.abs(), dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        GHG = GHG / scale
        evals, evecs = torch.linalg.eig(GHG)
        top = evals.real.argmax(dim=-1)
        w_T = evecs.gather(-1, top[:, None, None].expand(-1, GHG.shape[-1], 1)).squeeze(-1)
        w_T = w_T / (w_T.norm(dim=-1, keepdim=True) + 1e-9)
    w_T = w_T.detach().to(G_T.dtype)

    def absq(a, b):   # |a^H b|^2
        return (a.conj() * b).sum(-1).abs() ** 2

    g_nn = absq(h_n, w_n)                                    # = ||h_n||^2
    g_ff = absq(h_f, w_f)                                    # = ||h_f||^2
    gw   = torch.einsum("bmn,bn->bm", G_T, w_T)             # G_T w_T  (B, M)
    g_sT = (gw.abs() ** 2).sum(-1)                           # ||G_T w_T||^2

    return dict(g_nn=g_nn, g_ff=g_ff, g_sT=g_sT)


def rates_from_gains_oma(gains, t_alpha, P_tot, sigma2):
    t_n, t_f, t_T = t_alpha[:, 0], t_alpha[:, 1], t_alpha[:, 2]
    R_n = t_n * torch.log2(1.0 + gains['g_nn'] * P_tot / sigma2)
    R_f = t_f * torch.log2(1.0 + gains['g_ff'] * P_tot / sigma2)
    R_s = t_T * torch.log2(1.0 + gains['g_sT'] * P_tot / sigma2)
    return R_n, R_f, R_s


def make_features_oma(gains, R_th_c, R_th_s, P_tot, sigma2):
    """3 gain features (dB) + 2 QoS + 1 SNR budget = 6-dim input."""
    eps   = 1e-20
    g_db  = [10.0 * torch.log10(gains[k] + eps) for k in ('g_nn', 'g_ff', 'g_sT')]
    feats = torch.stack(g_db, dim=-1).float()
    B     = feats.shape[0]
    qos   = torch.tensor([R_th_c, R_th_s], dtype=torch.float32,
                         device=feats.device).expand(B, 2)
    snr_db = torch.full((B, 1), float(10.0 * np.log10(P_tot / sigma2)),
                        dtype=torch.float32, device=feats.device)
    return torch.cat([feats, qos, snr_db], dim=-1)


# ============================================================
# 3. Lagrangian loss — identical structure to NOMA version
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
# 4. Inner loop — K-step phi optimisation (same MAML structure)
# ============================================================
def inner_optimize(model, batch, cfg, args, create_meta_graph: bool):
    H_BR, h_RDn, h_RDf, h_RT, h_TR = batch
    B, N, _ = H_BR.shape
    device  = H_BR.device

    phi = (2.0 * np.pi) * torch.rand(B, N, device=device)
    phi.requires_grad_(True)

    for _ in range(args.inner_steps):
        gains   = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        feats   = make_features_oma(gains, cfg['R_th_c'], cfg['R_th_s'],
                                    cfg['P_tot'], cfg['sigma2'])
        t_alpha = model(feats)
        R_n, R_f, R_s = rates_from_gains_oma(gains, t_alpha, cfg['P_tot'], cfg['sigma2'])
        L_inner, _ = lagrangian_loss(
            R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
            R_th_n=args.R_th_n, R_th_f=args.R_th_f,
            lam_c=args.lam_c, lam_s=args.lam_s,
            lam_n=args.lam_n, lam_f=args.lam_f,
            w_c=args.w_c, w_s=args.w_s,
        )
        d_phi = torch.autograd.grad(L_inner, phi, create_graph=create_meta_graph)[0]
        phi   = phi - args.gamma_theta * d_phi

    gains   = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
    feats   = make_features_oma(gains, cfg['R_th_c'], cfg['R_th_s'],
                                cfg['P_tot'], cfg['sigma2'])
    t_alpha = model(feats)
    R_n, R_f, R_s = rates_from_gains_oma(gains, t_alpha, cfg['P_tot'], cfg['sigma2'])
    L_final, R_sum = lagrangian_loss(
        R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
        R_th_n=args.R_th_n, R_th_f=args.R_th_f,
        lam_c=args.lam_c, lam_s=args.lam_s,
        lam_n=args.lam_n, lam_f=args.lam_f,
    )
    return L_final, R_sum.mean(), R_n, R_f, R_s, phi, t_alpha


# ============================================================
# 5. Baseline — random phi + equal time (1/3, 1/3, 1/3)
# ============================================================
@torch.no_grad()
def baseline_eval(loader, device, cfg):
    N               = cfg['N']
    P_tot, sigma2   = cfg['P_tot'], cfg['sigma2']
    beta_T          = cfg['beta_T']
    R_th_c, R_th_s  = cfg['R_th_c'], cfg['R_th_s']
    t_eq = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32, device=device)
    R_sum_tot, qany_tot, n_tot = 0.0, 0.0, 0
    for batch in loader:
        H_BR, h_RDn, h_RDf, h_RT, h_TR = [b.to(device) for b in batch]
        B    = H_BR.shape[0]
        phi  = 2.0 * np.pi * torch.rand(B, N, device=device)
        gains = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T)
        t_alpha = t_eq.expand(B, 3)
        R_n, R_f, R_s = rates_from_gains_oma(gains, t_alpha, P_tot, sigma2)
        R_sum = 0.7 * (R_n + R_f) + 0.3 * R_s
        qany  = (((R_n + R_f) < R_th_c) | (R_s < R_th_s)).float()
        R_sum_tot += R_sum.sum().item()
        qany_tot  += qany.sum().item()
        n_tot     += B
    return R_sum_tot / n_tot, qany_tot / n_tot


# ============================================================
# 6. Training driver
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat",         default="ISAC_RIS_OMA_channels_v3_easy.mat")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch",       type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--inner_steps", type=int,   default=5)
    parser.add_argument("--gamma_theta", type=float, default=0.3)
    parser.add_argument("--lam_c",       type=float, default=2.0)
    parser.add_argument("--lam_s",       type=float, default=50.0)
    parser.add_argument("--lam_n",       type=float, default=15.0)
    parser.add_argument("--lam_f",       type=float, default=20.0)
    parser.add_argument("--R_th_n",      type=float, default=1.0)
    parser.add_argument("--R_th_f",      type=float, default=1.0)
    parser.add_argument("--w_c",         type=float, default=0.7)
    parser.add_argument("--w_s",         type=float, default=0.3)
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--seed",        type=int,   default=7)
    parser.add_argument("--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out",         default="policy_oma_maml_best.pt")
    parser.add_argument("--iter_log",    default=None)
    parser.add_argument("--P_tot_dBm",   type=float, default=None,
        help="Override P_tot (dBm). Channels are P-independent so overriding is sound.")
    args = parser.parse_args()
    if args.iter_log is None:
        args.iter_log = str(Path(args.out).with_suffix("")) + "_iters.npz"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

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
    full    = TensorDataset(*tensors)
    n_train = int(0.9 * len(full))
    n_val   = len(full) - n_train
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch)

    print(f"OMA-RIS MAML  |  N={cfg['N']}  M={cfg['M']}  "
          f"train={n_train}  val={n_val}")
    print(f"R_th_c={cfg['R_th_c']}  R_th_s={cfg['R_th_s']}  "
          f"P_tot={cfg['P_tot']:.3e}")
    print(f"inner_steps={args.inner_steps}  gamma_theta={args.gamma_theta}  "
          f"lr={args.lr}")
    print(f"lam (c/s/n/f) = {args.lam_c}/{args.lam_s}/{args.lam_n}/{args.lam_f}")

    R_base, q_base = baseline_eval(val_loader, device, cfg)
    print(f"[Baseline] random-phi + equal-time (1/3,1/3,1/3): "
          f"R={R_base:.4f}  qany={q_base:.4f}\n")

    model     = AlphaNet(in_dim=6, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iter_losses, iter_R = [], []

    def epoch_pass(loader, train_mode):
        model.train(train_mode)
        agg = dict(L=0, R=0, qany=0, qc=0, qs=0, n=0)
        for batch in loader:
            batch = tuple(b.to(device) for b in batch)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                L, R_sum, R_n, R_f, R_s, _, _ = inner_optimize(
                    model, batch, cfg, args, create_meta_graph=True)
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                iter_losses.append(L.item())
                iter_R.append(R_sum.item())
            else:
                with torch.enable_grad():
                    L, R_sum, R_n, R_f, R_s, _, _ = inner_optimize(
                        model, batch, cfg, args, create_meta_graph=False)
            B    = batch[0].size(0)
            qc   = ((R_n + R_f) < cfg['R_th_c']).float()
            qs   = (R_s < cfg['R_th_s']).float()
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
        va = epoch_pass(val_loader,   False)
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
    print(f"Baseline reference : R={R_base:.4f}  qany={q_base:.4f}")

    np.savez(args.iter_log,
             iter_losses=np.array(iter_losses, dtype=np.float32),
             iter_R=np.array(iter_R, dtype=np.float32),
             batch=args.batch, lr=args.lr,
             epochs=args.epochs, inner_steps=args.inner_steps)
    print(f"Per-iter log: {args.iter_log}  ({len(iter_losses)} steps)")


if __name__ == "__main__":
    main()

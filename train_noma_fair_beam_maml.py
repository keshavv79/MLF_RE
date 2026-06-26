"""
train_noma_fair_beam_maml.py
====================================================================
NOMA fair MAML trainer with a LEARNABLE common-beam split (fix #1) and
a stronger RIS inner loop (fix #3), aimed at improving NOMA sum-rate.

Difference vs train_noma_fair_maml.py:
  * AlphaNet now outputs 4 values: 3 power-split logits (softmax) + 1
    beam logit (sigmoid -> beta in (0,1)).
  * Common beam is w_c = beta*h_f_hat + (1-beta)*h_n_hat  (channels
    normalised first), learned per channel, instead of fixed 0.7/0.3.
    The policy can steer toward the rate-carrying near user when the
    far user has QoS slack -> R_n rises -> sum-rate rises.
  * Features are still computed from the fixed 0.7/0.3 reference beam,
    so the network input distribution (and the baseline) are unchanged
    -> the improvement metric isolates exactly the new capability.
  * inner_steps default 8, gamma_theta default 0.5 (stronger inner loop).

Self-contained: imports only load_dataset. Baseline = fixed beam +
fixed (0.20/0.55/0.25) alloc + random phi, identical to the original.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from train_dl_v3_easy import load_dataset


# ============================================================
# AlphaNet: 3 power logits (softmax) + 1 beam logit (sigmoid)
# ============================================================
class BeamAlphaNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        out = self.net(x)
        alpha = torch.softmax(out[:, :3], dim=-1)
        beta = torch.sigmoid(out[:, 3])
        return alpha, beta


# ============================================================
# Physics: core (beta-independent, eig once) + cheap beam gains
# ============================================================
def physics_core(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T):
    theta = torch.complex(torch.cos(phi), torch.sin(phi)).to(H_BR.dtype)
    TH = theta.unsqueeze(-1) * H_BR
    h_n = torch.einsum("bn,bnm->bm", h_RDn, TH)
    h_f = torch.einsum("bn,bnm->bm", h_RDf, TH)

    th_tr = theta * h_TR
    v1 = torch.einsum("bnm,bn->bm", H_BR.conj(), th_tr)
    v2 = torch.einsum("bn,bnm->bm", h_RT, TH)
    G_T = (beta_T ** 0.5) * v1.unsqueeze(-1) * v2.unsqueeze(-2)

    H_u = torch.stack([h_n, h_f], dim=-1)
    HuHu = H_u @ H_u.conj().transpose(-1, -2)
    with torch.no_grad():
        A = G_T.conj().transpose(-1, -2) @ G_T - 0.5 * HuHu
        A = 0.5 * (A + A.conj().transpose(-1, -2))
        scale = torch.amax(A.abs(), dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        A = A / scale
        evals, evecs = torch.linalg.eig(A)
        top = evals.real.argmax(dim=-1)
        w_T = evecs.gather(-1, top[:, None, None].expand(-1, A.shape[-1], 1)).squeeze(-1)
        w_T = w_T / (w_T.norm(dim=-1, keepdim=True) + 1e-9)
    w_T = w_T.detach().to(G_T.dtype)

    gw = torch.einsum("bmn,bn->bm", G_T, w_T)
    u_T = gw / (gw.norm(dim=-1, keepdim=True) + 1e-9)
    return dict(h_n=h_n, h_f=h_f, G_T=G_T, w_T=w_T, u_T=u_T)


def gains_from_beam(core, beta):
    """beta=None -> fixed 0.7/0.3 reference beam (for features & baseline).
       beta tensor (B,) -> learned beam on normalised channels."""
    h_n, h_f, G_T, w_T, u_T = core['h_n'], core['h_f'], core['G_T'], core['w_T'], core['u_T']
    if beta is None:
        w_c = 0.7 * h_f + 0.3 * h_n
    else:
        hf_hat = h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-9)
        hn_hat = h_n / (h_n.norm(dim=-1, keepdim=True) + 1e-9)
        b = beta.unsqueeze(-1)
        w_c = b * hf_hat + (1.0 - b) * hn_hat
    w_c = w_c / (w_c.norm(dim=-1, keepdim=True) + 1e-9)

    def absq(a, bb):
        return (a.conj() * bb).sum(-1).abs() ** 2

    def gscat(u, G, w):
        Gw = torch.einsum("bmn,bn->bm", G, w)
        return (u.conj() * Gw).sum(-1).abs() ** 2

    gff = absq(h_f, w_c); gfn = absq(h_f, w_c); gfT = absq(h_f, w_T)
    gnn = absq(h_n, w_c); gnT = absq(h_n, w_T); gsT = gscat(u_T, G_T, w_T)
    return dict(gff=gff, gfn=gfn, gfT=gfT, gnn=gnn, gnT=gnT, gsT=gsT)


def rates_from_gains(gains, alpha, P_tot, sigma2):
    a_n, a_f, a_T = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    Pn = a_n * P_tot; Pf = a_f * P_tot; PT = a_T * P_tot
    snr_f = (gains['gff'] * Pf) / (gains['gfn'] * Pn + gains['gfT'] * PT + sigma2)
    snr_n = (gains['gnn'] * Pn) / (gains['gnT'] * PT + sigma2)
    snr_s = (gains['gsT'] * PT) / sigma2
    return torch.log2(1 + snr_n), torch.log2(1 + snr_f), torch.log2(1 + snr_s)


def make_features(gains, R_th_c, R_th_s, P_tot, sigma2):
    eps = 1e-20
    g_db = [10.0 * torch.log10(gains[k] + eps)
            for k in ('gnn', 'gff', 'gsT', 'gfn', 'gfT', 'gnT')]
    feats = torch.stack(g_db, dim=-1).float()
    B = feats.shape[0]
    qos = torch.tensor([R_th_c, R_th_s], dtype=torch.float32, device=feats.device).expand(B, 2)
    snr_db = torch.full((B, 1), float(10.0 * np.log10(P_tot / sigma2)),
                        dtype=torch.float32, device=feats.device)
    return torch.cat([feats, qos, snr_db], dim=-1)


def lagrangian_loss(R_n, R_f, R_s, R_th_c, R_th_s, R_th_n=0.0, R_th_f=0.0,
                    w_c=0.7, w_s=0.3, lam_c=2.0, lam_s=50.0, lam_n=0.0, lam_f=0.0):
    R_sum = w_c * (R_n + R_f) + w_s * R_s
    pen_c = torch.relu(R_th_c - (R_n + R_f))
    pen_s = torch.relu(R_th_s - R_s)
    pen_n = torch.relu(R_th_n - R_n)
    pen_f = torch.relu(R_th_f - R_f)
    L = (-R_sum.mean() + lam_c * pen_c.mean() + lam_s * pen_s.mean()
         + lam_n * pen_n.mean() + lam_f * pen_f.mean())
    return L, R_sum


def inner_optimize(model, batch, cfg, args, create_meta_graph: bool):
    H_BR, h_RDn, h_RDf, h_RT, h_TR = batch
    B, N, _ = H_BR.shape
    device = H_BR.device
    phi = (2.0 * np.pi) * torch.rand(B, N, device=device)
    phi.requires_grad_(True)

    def forward(phi):
        core = physics_core(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, cfg['beta_T'])
        feats = make_features(gains_from_beam(core, None),
                              cfg['R_th_c'], cfg['R_th_s'], cfg['P_tot'], cfg['sigma2'])
        alpha, beta = model(feats)
        gains = gains_from_beam(core, beta)
        return rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2']) + (beta,)

    for _ in range(args.inner_steps):
        R_n, R_f, R_s, _ = forward(phi)
        L_inner, _ = lagrangian_loss(
            R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
            R_th_n=args.R_th_n, R_th_f=args.R_th_f,
            lam_c=args.lam_c, lam_s=args.lam_s, lam_n=args.lam_n, lam_f=args.lam_f,
            w_c=args.w_c, w_s=args.w_s)
        d_phi = torch.autograd.grad(L_inner, phi, create_graph=create_meta_graph)[0]
        phi = phi - args.gamma_theta * d_phi

    R_n, R_f, R_s, beta = forward(phi)
    L_final, R_sum = lagrangian_loss(
        R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
        R_th_n=args.R_th_n, R_th_f=args.R_th_f,
        lam_c=args.lam_c, lam_s=args.lam_s, lam_n=args.lam_n, lam_f=args.lam_f)
    return L_final, R_sum.mean(), R_n, R_f, R_s, beta


@torch.no_grad()
def baseline_eval(loader, device, cfg, a_baseline=(0.20, 0.55, 0.25)):
    N = cfg['N']; P_tot, sigma2, beta_T = cfg['P_tot'], cfg['sigma2'], cfg['beta_T']
    R_th_c, R_th_s = cfg['R_th_c'], cfg['R_th_s']
    R_tot = q_tot = n_tot = 0.0
    for batch in loader:
        H_BR, h_RDn, h_RDf, h_RT, h_TR = [b.to(device) for b in batch]
        B = H_BR.shape[0]
        phi = 2.0 * np.pi * torch.rand(B, N, device=device)
        core = physics_core(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T)
        gains = gains_from_beam(core, None)
        alpha = torch.tensor(a_baseline, dtype=torch.float32, device=device).expand(B, 3)
        R_n, R_f, R_s = rates_from_gains(gains, alpha, P_tot, sigma2)
        R_sum = 0.7 * (R_n + R_f) + 0.3 * R_s
        qany = (((R_n + R_f) < R_th_c) | (R_s < R_th_s)).float()
        R_tot += R_sum.sum().item(); q_tot += qany.sum().item(); n_tot += B
    return R_tot / n_tot, q_tot / n_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat",         default="ISAC_RIS_NOMA_channels_v3_fair.mat")
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch",       type=int,   default=128)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--inner_steps", type=int,   default=8)     # fix #3
    p.add_argument("--gamma_theta", type=float, default=0.5)   # fix #3
    p.add_argument("--lam_c",       type=float, default=2.0)
    p.add_argument("--lam_s",       type=float, default=50.0)
    p.add_argument("--lam_n",       type=float, default=15.0)
    p.add_argument("--lam_f",       type=float, default=20.0)
    p.add_argument("--R_th_n",      type=float, default=1.0)
    p.add_argument("--R_th_f",      type=float, default=1.0)
    p.add_argument("--w_c",         type=float, default=0.7)
    p.add_argument("--w_s",         type=float, default=0.3)
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--seed",        type=int,   default=7)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--out",         default="policy_noma_fair_beam_best.pt")
    p.add_argument("--P_tot_dBm",   type=float, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    ds = load_dataset(Path(args.mat))
    P_tot_eff = ds['P_tot']
    if args.P_tot_dBm is not None:
        P_tot_eff = 10.0 ** ((args.P_tot_dBm - 30.0) / 10.0)
    cfg = dict(N=ds['N'], M=ds['M'], P_tot=P_tot_eff, sigma2=ds['sigma2'],
               beta_T=ds['beta_T'], R_th_c=ds['R_th_c'], R_th_s=ds['R_th_s'])
    tensors = (torch.from_numpy(ds['H_BR']), torch.from_numpy(ds['h_RDn']),
               torch.from_numpy(ds['h_RDf']), torch.from_numpy(ds['h_RT']),
               torch.from_numpy(ds['h_TR']))
    full = TensorDataset(*tensors)
    n_train = int(0.9 * len(full)); n_val = len(full) - n_train
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    print(f"[BEAM] N={cfg['N']} M={cfg['M']} train={n_train} val={n_val} "
          f"inner_steps={args.inner_steps} gamma={args.gamma_theta}")
    R_base, q_base = baseline_eval(val_loader, device, cfg)
    print(f"[Baseline] fixed-beam + (0.20/0.55/0.25): R={R_base:.4f} qany={q_base:.4f}\n")

    model = BeamAlphaNet(in_dim=9, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def epoch_pass(loader, train_mode):
        model.train(train_mode)
        agg = dict(L=0, R=0, qany=0, qc=0, qs=0, beta=0, n=0)
        for batch in loader:
            batch = tuple(b.to(device) for b in batch)
            if train_mode:
                opt.zero_grad(set_to_none=True)
                L, R_sum, R_n, R_f, R_s, beta = inner_optimize(model, batch, cfg, args, False)
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            else:
                with torch.enable_grad():
                    L, R_sum, R_n, R_f, R_s, beta = inner_optimize(model, batch, cfg, args, False)
            B = batch[0].size(0)
            qc = ((R_n + R_f) < cfg['R_th_c']).float()
            qs = (R_s < cfg['R_th_s']).float()
            qany = ((qc + qs) > 0).float()
            agg['L'] += L.item() * B; agg['R'] += R_sum.item() * B
            agg['qc'] += qc.mean().item() * B; agg['qs'] += qs.mean().item() * B
            agg['qany'] += qany.mean().item() * B; agg['beta'] += beta.mean().item() * B
            agg['n'] += B
        return {k: v / agg['n'] for k, v in agg.items() if k != 'n'}

    print(f"{'ep':>3} | {'R_tr':>6} | {'R_val':>6} {'qany':>6} {'beta':>5}")
    best_score = -float("inf"); best_va = None
    for ep in range(1, args.epochs + 1):
        tr = epoch_pass(train_loader, True)
        va = epoch_pass(val_loader, False)
        print(f"{ep:3d} | {tr['R']:6.3f} | {va['R']:6.3f} {va['qany']:6.3f} {va['beta']:5.3f}")
        score = va['R'] - 5.0 * va['qany']
        if score > best_score:
            best_score = score; best_va = va
            torch.save({'state_dict': model.state_dict(), 'args': vars(args),
                        'cfg': cfg, 'val': va, 'baseline': {'R': R_base, 'qany': q_base}}, args.out)

    print(f"\n=== RESULT (best epoch) ===")
    print(f"Baseline : R={R_base:.4f}  qany={q_base:.4f}")
    print(f"BeamMAML : R={best_va['R']:.4f}  qany={best_va['qany']:.4f}  beta={best_va['beta']:.3f}")
    print(f"R improvement vs baseline: {100*(best_va['R']-R_base)/R_base:+.2f}%")


if __name__ == "__main__":
    main()

"""
sanity_check.py
====================================================================
Quick sanity check: run NOMA and OMA physics on the same random batch
and compare gains and rates.  No training — just one forward pass.

Usage (from OMA_new/):
    python sanity_check.py
    python sanity_check.py --mat ISAC_RIS_NOMA_channels_v3_easy.mat --N 256
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from train_dl_v3_easy import load_dataset

# ------------------------------------------------------------------ #
#  NOMA physics (verbatim from Anshul's train_dl_v3_easy_maml.py)    #
# ------------------------------------------------------------------ #
def physics_step_noma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T):
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

    w_c = h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-9)

    H_u = torch.stack([h_n, h_f], dim=-1)
    HuH = H_u.conj().transpose(-1, -2)
    HH = HuH @ H_u
    eye2 = torch.eye(2, dtype=cdtype, device=H_u.device).expand_as(HH)
    HH_inv = torch.linalg.inv(HH + 1e-6 * eye2)
    eyeM = torch.eye(M, dtype=cdtype, device=H_u.device).expand(B, M, M)
    P_null = eyeM - H_u @ HH_inv @ HuH

    A = P_null @ G_T.conj().transpose(-1, -2) @ G_T @ P_null
    A = 0.5 * (A + A.conj().transpose(-1, -2))
    _, eigvecs = torch.linalg.eigh(A)
    w_T = eigvecs[..., -1]
    w_T = w_T / (w_T.norm(dim=-1, keepdim=True) + 1e-9)

    gw = torch.einsum("bmn,bn->bm", G_T, w_T)
    u_T = gw / (gw.norm(dim=-1, keepdim=True) + 1e-9)

    def absq(a, b):
        return (a.conj() * b).sum(-1).abs() ** 2
    def gscat(u, G, w):
        Gw = torch.einsum("bmn,bn->bm", G, w)
        return (u.conj() * Gw).sum(-1).abs() ** 2

    gff = absq(h_f, w_c)
    gfn = absq(h_f, w_c)
    gfT = absq(h_f, w_T)
    gnn = absq(h_n, w_c)
    gnT = absq(h_n, w_T)
    gsT = gscat(u_T, G_T, w_T)
    return dict(gff=gff, gfn=gfn, gfT=gfT, gnn=gnn, gnT=gnT, gsT=gsT)


def rates_noma(gains, alpha, P_tot, sigma2):
    a_n, a_f, a_T = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    Pn = a_n * P_tot; Pf = a_f * P_tot; PT = a_T * P_tot
    snr_f = (gains['gff'] * Pf) / (gains['gfn'] * Pn + gains['gfT'] * PT + sigma2)
    snr_n = (gains['gnn'] * Pn) / (gains['gnT'] * PT + sigma2)
    snr_s = (gains['gsT'] * PT) / sigma2
    return torch.log2(1 + snr_n), torch.log2(1 + snr_f), torch.log2(1 + snr_s)


# ------------------------------------------------------------------ #
#  OMA physics (from train_dl_oma_maml.py)                           #
# ------------------------------------------------------------------ #
def physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T):
    theta = torch.complex(torch.cos(phi), torch.sin(phi)).to(H_BR.dtype)
    B, N, M = H_BR.shape

    TH  = theta.unsqueeze(-1) * H_BR
    h_n = torch.einsum("bn,bnm->bm", h_RDn, TH)
    h_f = torch.einsum("bn,bnm->bm", h_RDf, TH)

    th_tr = theta * h_TR
    v1    = torch.einsum("bnm,bn->bm", H_BR.conj(), th_tr)
    v2    = torch.einsum("bn,bnm->bm", h_RT, TH)
    G_T   = (beta_T ** 0.5) * v1.unsqueeze(-1) * v2.unsqueeze(-2)

    w_n = h_n / (h_n.norm(dim=-1, keepdim=True) + 1e-9)
    w_f = h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-9)

    GHG = G_T.conj().transpose(-1, -2) @ G_T
    GHG = 0.5 * (GHG + GHG.conj().transpose(-1, -2))
    _, eigvecs = torch.linalg.eigh(GHG)
    w_T = eigvecs[..., -1]
    w_T = w_T / (w_T.norm(dim=-1, keepdim=True) + 1e-9)

    def absq(a, b):
        return (a.conj() * b).sum(-1).abs() ** 2

    g_nn = absq(h_n, w_n)
    g_ff = absq(h_f, w_f)
    gw   = torch.einsum("bmn,bn->bm", G_T, w_T)
    g_sT = (gw.abs() ** 2).sum(-1)
    return dict(g_nn=g_nn, g_ff=g_ff, g_sT=g_sT)


def rates_oma(gains, t_alpha, P_tot, sigma2):
    t_n, t_f, t_T = t_alpha[:, 0], t_alpha[:, 1], t_alpha[:, 2]
    R_n = t_n * torch.log2(1.0 + gains['g_nn'] * P_tot / sigma2)
    R_f = t_f * torch.log2(1.0 + gains['g_ff'] * P_tot / sigma2)
    R_s = t_T * torch.log2(1.0 + gains['g_sT'] * P_tot / sigma2)
    return R_n, R_f, R_s


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', default='ISAC_RIS_NOMA_channels_v3_easy.mat')
    ap.add_argument('--N',   type=int, default=256, help='batch size for the check')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = load_dataset(Path(args.mat))
    P_tot  = ds['P_tot']
    sigma2 = ds['sigma2']
    beta_T = ds['beta_T']
    N      = ds['N']
    M      = ds['M']

    print(f"Dataset: N={N}  M={M}  P_tot={P_tot:.3e} W  sigma2={sigma2:.3e}")
    print(f"R_th_c={ds['R_th_c']}  R_th_s={ds['R_th_s']}")
    snr_dB = 10 * np.log10(P_tot / sigma2)
    print(f"SNR budget = {snr_dB:.1f} dB\n")

    B = min(args.N, ds['H_BR'].shape[0])
    idx  = np.random.choice(ds['H_BR'].shape[0], B, replace=False)
    H_BR  = torch.from_numpy(ds['H_BR'][idx])
    h_RDn = torch.from_numpy(ds['h_RDn'][idx])
    h_RDf = torch.from_numpy(ds['h_RDf'][idx])
    h_RT  = torch.from_numpy(ds['h_RT'][idx])
    h_TR  = torch.from_numpy(ds['h_TR'][idx])

    phi = 2.0 * np.pi * torch.rand(B, N)   # same random phi for both

    # ---- NOMA: equal power allocation (1/3 each) ----
    # alpha_eq = torch.full((B, 3), 1/3)
    alpha_eq = torch.tensor([0.1, 0.65, 0.25]).expand(B, 3)

    with torch.no_grad():
        gn = physics_step_noma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T)
        Rn_n, Rn_f, Rn_s = rates_noma(gn, alpha_eq, P_tot, sigma2)
    Rn_sum = 0.7 * (Rn_n + Rn_f) + 0.3 * Rn_s

    # ---- OMA: equal time allocation (1/3 each) ----
    # t_eq = torch.full((B, 3), 1/3)
    t_eq = torch.tensor([0.1, 0.65, 0.25]).expand(B, 3)
    with torch.no_grad():
        go = physics_step_oma(phi, H_BR, h_RDn, h_RDf, h_RT, h_TR, beta_T)
        Ro_n, Ro_f, Ro_s = rates_oma(go, t_eq, P_tot, sigma2)
    Ro_sum = 0.7 * (Ro_n + Ro_f) + 0.3 * Ro_s

    def db(x):  return 10 * np.log10(float(x) + 1e-20)

    print("=" * 60)
    print(f"{'':30s}  {'NOMA':>10}  {'OMA':>10}")
    print("=" * 60)
    print(f"{'Gains (mean linear)':30s}")
    print(f"  {'g_nn / gnn':28s}  {gn['gnn'].mean().item():10.4e}  {go['g_nn'].mean().item():10.4e}")
    print(f"  {'g_ff / gff':28s}  {gn['gff'].mean().item():10.4e}  {go['g_ff'].mean().item():10.4e}")
    print(f"  {'g_sT / gsT':28s}  {gn['gsT'].mean().item():10.4e}  {go['g_sT'].mean().item():10.4e}")
    print()
    print(f"{'Gains (mean dB)':30s}")
    print(f"  {'g_nn (dB)':28s}  {db(gn['gnn'].mean()):10.2f}  {db(go['g_nn'].mean()):10.2f}")
    print(f"  {'g_ff (dB)':28s}  {db(gn['gff'].mean()):10.2f}  {db(go['g_ff'].mean()):10.2f}")
    print(f"  {'g_sT (dB)':28s}  {db(gn['gsT'].mean()):10.2f}  {db(go['g_sT'].mean()):10.2f}")
    print()
    print(f"{'Rates (mean bits/s/Hz, equal alloc)':30s}")
    print(f"  {'R_n':28s}  {Rn_n.mean().item():10.4f}  {Ro_n.mean().item():10.4f}")
    print(f"  {'R_f':28s}  {Rn_f.mean().item():10.4f}  {Ro_f.mean().item():10.4f}")
    print(f"  {'R_s':28s}  {Rn_s.mean().item():10.4f}  {Ro_s.mean().item():10.4f}")
    print(f"  {'R_n + R_f':28s}  {(Rn_n+Rn_f).mean().item():10.4f}  {(Ro_n+Ro_f).mean().item():10.4f}")
    print(f"  {'R_sum (0.7*comm + 0.3*sens)':28s}  {Rn_sum.mean().item():10.4f}  {Ro_sum.mean().item():10.4f}")
    print("=" * 60)

    # QoS check
    qc_n = ((Rn_n + Rn_f) < ds['R_th_c']).float().mean().item()
    qs_n = (Rn_s < ds['R_th_s']).float().mean().item()
    qc_o = ((Ro_n + Ro_f) < ds['R_th_c']).float().mean().item()
    qs_o = (Ro_s < ds['R_th_s']).float().mean().item()
    print(f"\n{'QoS violation rate (equal alloc, random phi)':30s}")
    print(f"  {'comm (R_n+R_f < R_th_c)':28s}  {qc_n:10.3f}  {qc_o:10.3f}")
    print(f"  {'sens (R_s < R_th_s)':28s}  {qs_n:10.3f}  {qs_o:10.3f}")
    print()
    print("Note: high violation rate with random phi + equal alloc is expected —")
    print("      training (MAML inner loop) is what optimises phi and allocation.")

    # Quick cross-check: g_nn == ||h_n||^2 in OMA (should hold by construction)
    hn_norm_sq = (torch.einsum("bn,bnm->bm", h_RDn,
                               torch.complex(torch.cos(phi), torch.sin(phi)).unsqueeze(-1) * H_BR
                               ).norm(dim=-1) ** 2)
    err = (go['g_nn'] - hn_norm_sq).abs().max().item()
    print(f"\n[self-check] max |g_nn - ||h_n||^2| = {err:.2e}  (should be ~0)")


if __name__ == '__main__':
    main()

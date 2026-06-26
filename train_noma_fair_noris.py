"""
train_noris.py
====================================================================
Single-shot DL trainer for the No-RIS MISO ISAC-NOMA branch.
AlphaNet only: predicts power split (a_n, a_f, a_T) on the simplex
from direct-channel features. No inner loop, no phi -- no RIS to
optimize. Beamformers (cluster MRT + null-space sensing) are
closed-form on the direct channels and identical in structure to
the RIS branch in train_dl_v3_easy_maml.py.

Inputs : paper_recreation/ISAC_NOMA_channels_noris.mat
         (produced by paper_recreation/gen_noris_dataset.m)
Output : <out>.pt  +  <out>_iters.npz
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'N8'))
from train_noma_fair_maml import AlphaNet, lagrangian_loss


# ============================================================
# 1. Dataset loading (matches v3_easy MATLAB -v7.3 / HDF5 layout)
# ============================================================
def _to_complex(ds):
    arr = ds[...]
    if arr.dtype.names is not None and "real" in arr.dtype.names:
        return arr["real"] + 1j * arr["imag"]
    return arr.astype(np.complex128)


def load_noris_dataset(mat_path: Path):
    with h5py.File(mat_path, "r") as f:
        h_BDn = _to_complex(f["h_BDn_all"])
        h_BDf = _to_complex(f["h_BDf_all"])
        h_BT  = _to_complex(f["h_BT_all"])

        def scalar(name):
            return float(np.array(f[name]).squeeze())

        M      = int(scalar("M"))
        P_tot  = scalar("P_tot")
        sigma2 = scalar("sigma2")
        beta_T = scalar("beta_T")
        R_th_c = scalar("R_th_c")
        R_th_s = scalar("R_th_s")

    # MATLAB (M, 1, num) -> numpy (num, 1, M) -> (num, M)
    h_BDn = h_BDn.reshape(-1, M).astype(np.complex64)
    h_BDf = h_BDf.reshape(-1, M).astype(np.complex64)
    h_BT  = h_BT.reshape(-1, M).astype(np.complex64)
    return dict(h_BDn=h_BDn, h_BDf=h_BDf, h_BT=h_BT,
                M=M, P_tot=P_tot, sigma2=sigma2, beta_T=beta_T,
                R_th_c=R_th_c, R_th_s=R_th_s)


# ============================================================
# 2. Physics: direct-channel gains + closed-form beamformers
# ============================================================
def noris_gains(h_n, h_f, h_BT, beta_T):
    """
    h_n, h_f, h_BT: (B, M) complex.
    Returns the same six effective scalar gains used in
    train_dl_v3_easy_maml.rates_from_gains.
    """
    B, M = h_n.shape
    cdtype = h_n.dtype

    # Monostatic round-trip sensing channel: G_T = sqrt(beta_T) * h_BT h_BT^T
    G_T = (beta_T ** 0.5) * h_BT.unsqueeze(-1) * h_BT.unsqueeze(-2)

    # FAIR weighted common beamformer (w_n = w_f = 0.7 h_f + 0.3 h_n)
    w_c = 0.7 * h_f + 0.3 * h_n
    w_c = w_c / (w_c.norm(dim=-1, keepdim=True) + 1e-9)

    # FAIR soft sensing nulling: A = G^H G - lambda * H_u H_u^H  (lambda=0.5).
    # w_T = top eigenvector (closed form, no grad through the phase-ambiguous eigh).
    H_u = torch.stack([h_n, h_f], dim=-1)            # (B, M, 2)
    HuHu = H_u @ H_u.conj().transpose(-1, -2)        # (B, M, M)
    with torch.no_grad():
        A = G_T.conj().transpose(-1, -2) @ G_T - 0.5 * HuHu
        A = 0.5 * (A + A.conj().transpose(-1, -2))
        # general eig (matches MATLAB; robust where eigh fails). Normalise first.
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

    return dict(
        gff=absq(h_f, w_c),
        gfn=absq(h_f, w_c),
        gfT=absq(h_f, w_T),
        gnn=absq(h_n, w_c),
        gnT=absq(h_n, w_T),
        gsT=gscat(u_T, G_T, w_T),
    )


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
# 3. Train / eval driver
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat",    default="ISAC_NOMA_channels_fair_noris.mat")
    parser.add_argument("--epochs", type=int,   default=60)
    parser.add_argument("--batch",  type=int,   default=128)
    parser.add_argument("--lr",     type=float, default=5e-4)
    parser.add_argument("--lam_c",  type=float, default=2.0)
    parser.add_argument("--lam_s",  type=float, default=50.0)
    parser.add_argument("--lam_n",  type=float, default=15.0)
    parser.add_argument("--lam_f",  type=float, default=20.0)
    parser.add_argument("--R_th_n", type=float, default=1.0)
    parser.add_argument("--R_th_f", type=float, default=1.0)
    parser.add_argument("--w_c",    type=float, default=0.7)
    parser.add_argument("--w_s",    type=float, default=0.3)
    parser.add_argument("--hidden", type=int,   default=256)
    parser.add_argument("--seed",   type=int,   default=7)
    parser.add_argument("--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out",      default="policy_noma_fair_noris_best.pt")
    parser.add_argument("--iter_log", default=None)
    parser.add_argument("--P_tot_dBm", type=float, default=None,
                        help="Override P_tot (dBm). If unset, uses value from .mat.")
    args = parser.parse_args()
    if args.iter_log is None:
        args.iter_log = str(Path(args.out).with_suffix("")) + "_iters.npz"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    ds = load_noris_dataset(Path(args.mat))
    P_tot_eff = ds['P_tot']
    if args.P_tot_dBm is not None:
        P_tot_eff = 10.0 ** ((args.P_tot_dBm - 30.0) / 10.0)
        print(f"[override] P_tot {ds['P_tot']:.3e} -> {P_tot_eff:.3e}  "
              f"({args.P_tot_dBm:.1f} dBm)")
    cfg = dict(M=ds['M'], P_tot=P_tot_eff, sigma2=ds['sigma2'],
               beta_T=ds['beta_T'], R_th_c=ds['R_th_c'], R_th_s=ds['R_th_s'])
    tensors = (torch.from_numpy(ds['h_BDn']),
               torch.from_numpy(ds['h_BDf']),
               torch.from_numpy(ds['h_BT']))
    full = TensorDataset(*tensors)
    n_train = int(0.9 * len(full))
    n_val = len(full) - n_train
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    print(f"Samples train={n_train} val={n_val} | M={cfg['M']}")
    print(f"R_th_c={cfg['R_th_c']}  R_th_s={cfg['R_th_s']}  P_tot={cfg['P_tot']:.3e}")
    print(f"SINGLE-SHOT DL (no RIS, no inner loop)  outer_lr={args.lr}  batch={args.batch}")
    print(f"lam (c/s/n/f) = {args.lam_c}/{args.lam_s}/{args.lam_n}/{args.lam_f}\n")

    model = AlphaNet(in_dim=9, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iter_losses = []
    iter_R = []

    def forward_pass(batch):
        h_n, h_f, h_BT = [b.to(device) for b in batch]
        gains = noris_gains(h_n, h_f, h_BT, cfg['beta_T'])
        feats = make_features(gains, cfg['R_th_c'], cfg['R_th_s'],
                              cfg['P_tot'], cfg['sigma2'])
        alpha = model(feats)
        R_n, R_f, R_s = rates_from_gains(gains, alpha, cfg['P_tot'], cfg['sigma2'])
        L, R_sum = lagrangian_loss(
            R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
            R_th_n=args.R_th_n, R_th_f=args.R_th_f,
            lam_c=args.lam_c, lam_s=args.lam_s,
            lam_n=args.lam_n, lam_f=args.lam_f,
            w_c=args.w_c, w_s=args.w_s,
        )
        return L, R_sum, R_n, R_f, R_s

    def epoch_pass(loader, train_mode):
        model.train(train_mode)
        agg = dict(L=0, R=0, qany=0, qc=0, qs=0, n=0)
        for batch in loader:
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                L, R_sum, R_n, R_f, R_s = forward_pass(batch)
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                iter_losses.append(L.item())
                iter_R.append(R_sum.mean().item())
            else:
                with torch.no_grad():
                    L, R_sum, R_n, R_f, R_s = forward_pass(batch)
            B = batch[0].size(0)
            qc = ((R_n + R_f) < cfg['R_th_c']).float()
            qs = (R_s < cfg['R_th_s']).float()
            qany = ((qc + qs) > 0).float()
            agg['L']    += L.item()         * B
            agg['R']    += R_sum.mean().item() * B
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
            }, args.out)

    print(f"\nBest score (R - 5*qany) = {best_score:.4f}  ->  {args.out}")
    np.savez(args.iter_log,
             iter_losses=np.array(iter_losses, dtype=np.float32),
             iter_R=np.array(iter_R, dtype=np.float32),
             batch=args.batch, lr=args.lr, epochs=args.epochs)
    print(f"Per-iter log saved: {args.iter_log}  ({len(iter_losses)} steps)")


if __name__ == "__main__":
    main()

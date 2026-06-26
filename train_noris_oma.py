"""
train_noris_oma.py
====================================================================
Single-shot DL trainer for OMA-TDMA, No-RIS branch.
AlphaNet predicts time fractions (t_n, t_f, t_T) on the simplex.
No inner loop — no RIS to optimise.

Input : no_ris/ISAC_NOMA_channels_noris.mat
Output: <out>.pt
"""

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(str(Path(__file__).resolve().parent))
from train_dl_oma_maml import (AlphaNet, rates_from_gains_oma,
                                make_features_oma, lagrangian_loss)


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
        def scalar(name): return float(np.array(f[name]).squeeze())
        M      = int(scalar("M"))
        P_tot  = scalar("P_tot")
        sigma2 = scalar("sigma2")
        beta_T = scalar("beta_T")
        R_th_c = scalar("R_th_c")
        R_th_s = scalar("R_th_s")
    h_BDn = h_BDn.reshape(-1, M).astype(np.complex64)
    h_BDf = h_BDf.reshape(-1, M).astype(np.complex64)
    h_BT  = h_BT.reshape(-1, M).astype(np.complex64)
    return dict(h_BDn=h_BDn, h_BDf=h_BDf, h_BT=h_BT,
                M=M, P_tot=P_tot, sigma2=sigma2, beta_T=beta_T,
                R_th_c=R_th_c, R_th_s=R_th_s)


def noris_gains_oma(h_n, h_f, h_BT, beta_T):
    """Per-user MRT + optimal sensing BF. Returns g_nn, g_ff, g_sT."""
    w_n = h_n / (h_n.norm(dim=-1, keepdim=True) + 1e-9)
    w_f = h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-9)

    # Monostatic round-trip channel (rank-1): G_T = sqrt(beta_T) * h_BT * h_BT^T
    G_T = (beta_T ** 0.5) * h_BT.unsqueeze(-1) * h_BT.unsqueeze(-2)  # (B, M, M)
    with torch.no_grad():
        GHG = G_T.conj().transpose(-1, -2) @ G_T
        GHG = 0.5 * (GHG + GHG.conj().transpose(-1, -2))
        scale = torch.amax(GHG.abs(), dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        GHG = GHG / scale
        evals, evecs = torch.linalg.eig(GHG)
        top = evals.real.argmax(dim=-1)
        w_T = evecs.gather(-1, top[:, None, None].expand(-1, GHG.shape[-1], 1)).squeeze(-1)
        w_T = w_T / (w_T.norm(dim=-1, keepdim=True) + 1e-9)
    w_T = w_T.detach().to(G_T.dtype)

    def absq(a, b): return (a.conj() * b).sum(-1).abs() ** 2
    g_nn = absq(h_n, w_n)
    g_ff = absq(h_f, w_f)
    gw   = torch.einsum("bmn,bn->bm", G_T, w_T)
    g_sT = (gw.abs() ** 2).sum(-1)
    return dict(g_nn=g_nn, g_ff=g_ff, g_sT=g_sT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat",       default="no_ris/ISAC_OMA_channels_noris.mat")
    parser.add_argument("--epochs",    type=int,   default=60)
    parser.add_argument("--batch",     type=int,   default=128)
    parser.add_argument("--lr",        type=float, default=5e-4)
    parser.add_argument("--lam_c",     type=float, default=2.0)
    parser.add_argument("--lam_s",     type=float, default=50.0)
    parser.add_argument("--lam_n",     type=float, default=15.0)
    parser.add_argument("--lam_f",     type=float, default=20.0)
    parser.add_argument("--R_th_n",    type=float, default=1.0)
    parser.add_argument("--R_th_f",    type=float, default=1.0)
    parser.add_argument("--w_c",       type=float, default=0.7)
    parser.add_argument("--w_s",       type=float, default=0.3)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--seed",      type=int,   default=7)
    parser.add_argument("--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out",       default="no_ris/policy_noris_oma_best.pt")
    parser.add_argument("--P_tot_dBm", type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    ds = load_noris_dataset(Path(args.mat))
    P_tot_eff = ds['P_tot']
    if args.P_tot_dBm is not None:
        P_tot_eff = 10.0 ** ((args.P_tot_dBm - 30.0) / 10.0)
        print(f"[override] P_tot {ds['P_tot']:.3e} -> {P_tot_eff:.3e}  "
              f"({args.P_tot_dBm:.1f} dBm)")
    cfg = dict(P_tot=P_tot_eff, sigma2=ds['sigma2'], beta_T=ds['beta_T'],
               R_th_c=ds['R_th_c'], R_th_s=ds['R_th_s'])

    tensors = (torch.from_numpy(ds['h_BDn']),
               torch.from_numpy(ds['h_BDf']),
               torch.from_numpy(ds['h_BT']))
    full    = TensorDataset(*tensors)
    n_train = int(0.9 * len(full))
    n_val   = len(full) - n_train
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch)

    print(f"OMA no-RIS  |  M={ds['M']}  train={n_train}  val={n_val}")
    print(f"R_th_c={cfg['R_th_c']}  R_th_s={cfg['R_th_s']}  "
          f"P_tot={cfg['P_tot']:.3e}")

    model     = AlphaNet(in_dim=6, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iter_losses, iter_R = [], []

    def epoch_pass(loader, train_mode):
        model.train(train_mode)
        agg = dict(L=0, R=0, qany=0, qc=0, qs=0, n=0)
        for h_n_b, h_f_b, h_BT_b in loader:
            h_n_b  = h_n_b.to(device)
            h_f_b  = h_f_b.to(device)
            h_BT_b = h_BT_b.to(device)
            B      = h_n_b.shape[0]
            gains  = noris_gains_oma(h_n_b, h_f_b, h_BT_b, cfg['beta_T'])
            feats  = make_features_oma(gains, cfg['R_th_c'], cfg['R_th_s'],
                                       cfg['P_tot'], cfg['sigma2'])
            t_alpha = model(feats)
            R_n, R_f, R_s = rates_from_gains_oma(gains, t_alpha,
                                                  cfg['P_tot'], cfg['sigma2'])
            L, R_sum = lagrangian_loss(
                R_n, R_f, R_s, cfg['R_th_c'], cfg['R_th_s'],
                R_th_n=args.R_th_n, R_th_f=args.R_th_f,
                lam_c=args.lam_c, lam_s=args.lam_s,
                lam_n=args.lam_n, lam_f=args.lam_f,
                w_c=args.w_c, w_s=args.w_s,
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                iter_losses.append(L.item())
                iter_R.append(R_sum.mean().item())
            qc   = ((R_n + R_f) < cfg['R_th_c']).float()
            qs   = (R_s < cfg['R_th_s']).float()
            qany = ((qc + qs) > 0).float()
            agg['L']    += L.item() * B
            agg['R']    += R_sum.mean().item() * B
            agg['qc']   += qc.mean().item() * B
            agg['qs']   += qs.mean().item() * B
            agg['qany'] += qany.mean().item() * B
            agg['n']    += B
        return {k: v / agg['n'] for k, v in agg.items() if k != 'n'}

    print(f"{'epoch':>5} | {'L_tr':>7} {'R_tr':>6} | "
          f"{'L_val':>7} {'R_val':>6} {'qany_v':>7}")
    best_score = -float("inf")
    for ep in range(1, args.epochs + 1):
        tr = epoch_pass(train_loader, True)
        va = epoch_pass(val_loader,   False)
        print(f"{ep:5d} | {tr['L']:7.3f} {tr['R']:6.3f} | "
              f"{va['L']:7.3f} {va['R']:6.3f} {va['qany']:7.3f}")
        score = va['R'] - 5.0 * va['qany']
        if score > best_score:
            best_score = score
            torch.save({
                'state_dict': model.state_dict(),
                'args': vars(args),
                'cfg': cfg,
                'val': va,
            }, args.out)
    print(f"\nBest score = {best_score:.4f}  ->  {args.out}")
    iter_log = str(Path(args.out).with_suffix("")) + "_iters.npz"
    np.savez(iter_log, iter_losses=np.array(iter_losses, dtype=np.float32),
             iter_R=np.array(iter_R, dtype=np.float32),
             batch=args.batch, lr=args.lr, epochs=args.epochs)
    print(f"Per-iter log saved: {iter_log}  ({len(iter_losses)} steps)")


if __name__ == "__main__":
    main()

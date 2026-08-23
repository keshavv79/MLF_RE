"""
train_dl_v3_easy.py
====================================================================
Joint DL optimisation of RIS phase shifts (Theta) and NOMA power
allocation (a_n, a_f, a_T) for the MISO RIS-ISAC-NOMA v3-easy dataset.

Pipeline
--------
1. Load ISAC_RIS_NOMA_channels_v3_easy.mat (output of miso_isac_noma_v3_easy.m).
2. Policy network: channel -> (N continuous phases, 3 power coeffs).
3. Differentiable physics forward (port of the MATLAB SINR/rate code):
       cascaded MISO channels -> cluster MRT comm BF + null-space sensing BF
       -> NOMA SIC SINRs -> rates.
4. Loss: -E[R_DL_sum] + lam_c * relu(R_th_c - (R_n+R_f)) + lam_s * relu(R_th_s - R_s)
5. Track: weighted sum-rate, Pr(qos_viol_any).  Save best policy.

Beamforming choice
------------------
Following v3-easy.m, w_n = w_f = h_f / ||h_f|| (cluster MRT on far user,
preserves NOMA SIC) and w_T = top eigvec of P_null G_T^H G_T P_null with
P_null = projector onto null{[h_n, h_f]}.  Both are closed-form,
differentiable in (Theta, channels), so backprop flows cleanly through
them.  The policy only learns (Theta, power); beamformers are pinned.

Prereq
------
1. Run miso_isac_noma_v3_easy.m in MATLAB to produce the .mat file.
2. pip install torch numpy h5py
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ============================================================
# 1. Dataset loading  (MATLAB -v7.3 = HDF5; complex stored as compound)
# ============================================================
def _to_complex(ds):
    arr = ds[...]
    if arr.dtype.names is not None and "real" in arr.dtype.names:
        return arr["real"] + 1j * arr["imag"]
    return arr.astype(np.complex128)


def load_dataset(mat_path: Path):
    with h5py.File(mat_path, "r") as f:
        H_BR  = _to_complex(f["H_BR_all"])    # (num, M, N) after HDF5 dim reversal
        h_RDn = _to_complex(f["h_RDn_all"])
        h_RDf = _to_complex(f["h_RDf_all"])
        h_RT  = _to_complex(f["h_RT_all"])
        h_TR  = _to_complex(f["h_TR_all"])

        def scalar(name):
            return float(np.array(f[name]).squeeze())

        N      = int(scalar("N"))
        M      = int(scalar("M"))
        P_tot  = scalar("P_tot")
        sigma2 = scalar("sigma2")
        beta_T = scalar("beta_T")
        R_th_c = scalar("R_th_c")
        R_th_s = scalar("R_th_s")

    # MATLAB (N, M, num) -> numpy (num, M, N) -> (num, N, M)
    H_BR  = np.transpose(H_BR, (0, 2, 1)).astype(np.complex64)
    h_RDn = h_RDn.reshape(-1, N).astype(np.complex64)
    h_RDf = h_RDf.reshape(-1, N).astype(np.complex64)
    h_RT  = h_RT.reshape(-1, N).astype(np.complex64)
    h_TR  = h_TR.reshape(-1, N).astype(np.complex64)

    return dict(
        H_BR=H_BR, h_RDn=h_RDn, h_RDf=h_RDf, h_RT=h_RT, h_TR=h_TR,
        N=N, M=M, P_tot=P_tot, sigma2=sigma2, beta_T=beta_T,
        R_th_c=R_th_c, R_th_s=R_th_s,
    )


# ============================================================
# 2. Policy network
# ============================================================
class PolicyNet(nn.Module):
    """
    Input  : flattened real features of (H_BR, h_RDn, h_RDf, h_RT, h_TR)
    Output : (theta, alpha)
        theta  : (B, N) complex on the unit circle      (RIS diag entries)
        alpha  : (B, 3) softmax-normalised power alloc  (a_n, a_f, a_T)
    """

    def __init__(self, N: int, M: int, hidden: int = 256):
        super().__init__()
        self.N = N
        self.M = M
        in_dim = 2 * (N * M + 4 * N)   # real+imag of all channel pieces
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # 2N outputs: (cos, sin) pair per phase, normalised to unit circle
        self.phase_head = nn.Linear(hidden, 2 * N)
        # 3 outputs: softmax -> simplex
        self.power_head = nn.Linear(hidden, 3)

    def forward(self, feat):
        h = self.backbone(feat)
        cs = self.phase_head(h).view(-1, self.N, 2)
        cs = cs / (cs.norm(dim=-1, keepdim=True) + 1e-9)
        theta = torch.complex(cs[..., 0].contiguous(),
                              cs[..., 1].contiguous())          # (B, N)
        alpha = torch.softmax(self.power_head(h), dim=-1)        # (B, 3)
        return theta, alpha


def channels_to_features(H_BR, h_RDn, h_RDf, h_RT, h_TR):
    def cc(x):
        return torch.cat([x.real, x.imag], dim=-1)
    B = H_BR.shape[0]
    return torch.cat([
        cc(H_BR.reshape(B, -1)),
        cc(h_RDn), cc(h_RDf), cc(h_RT), cc(h_TR),
    ], dim=-1).float()


# ============================================================
# 3. Differentiable forward (port of the MATLAB physics)
# ============================================================
def compute_rates(theta, alpha,
                  H_BR, h_RDn, h_RDf, h_RT, h_TR,
                  P_tot, sigma2, beta_T):
    """
    theta : (B, N) complex64  - diag entries of Theta
    alpha : (B, 3) float      - (a_n, a_f, a_T), sums to 1
    H_BR  : (B, N, M)
    h_*   : (B, N)
    """
    B, N, M = H_BR.shape
    cdtype = H_BR.dtype                    # complex64

    # Theta @ H_BR  via broadcasting (Theta is diagonal)
    TH = theta.unsqueeze(-1) * H_BR        # (B, N, M)

    # h_k = (h_RDk^T Theta H_BR)^T   ->  (B, M)
    # Equivalent to: H_BR^T @ (theta * h_RDk)
    h_n = torch.einsum("bn,bnm->bm", h_RDn, TH)
    h_f = torch.einsum("bn,bnm->bm", h_RDf, TH)

    # ---- Sensing channel G_T  (M x M) ----
    #   G_T = sqrt(beta_T) * (H_BR^H Theta h_TR) * (h_RT^T Theta H_BR)
    # First factor (M,): conj(H_BR)^T @ (theta * h_TR)
    th_tr = theta * h_TR                                       # (B, N)
    v1 = torch.einsum("bnm,bn->bm", H_BR.conj(), th_tr)        # (B, M)
    # Second factor (M,): h_RT^T @ TH
    v2 = torch.einsum("bn,bnm->bm", h_RT, TH)                  # (B, M)
    G_T = (beta_T ** 0.5) * v1.unsqueeze(-1) * v2.unsqueeze(-2)  # (B, M, M)

    # ---- Comm beamformer (cluster MRT on far user) ----
    w_c = h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-9)
    w_n = w_c
    w_f = w_c

    # ---- Sensing beamformer: top eigvec of P_null G_T^H G_T P_null ----
    H_u = torch.stack([h_n, h_f], dim=-1)                      # (B, M, 2)
    HuH = H_u.conj().transpose(-1, -2)                         # (B, 2, M)
    HH = HuH @ H_u                                             # (B, 2, 2)
    eye2 = torch.eye(2, dtype=cdtype, device=H_u.device).expand_as(HH)
    HH_inv = torch.linalg.inv(HH + 1e-6 * eye2)
    eyeM = torch.eye(M, dtype=cdtype, device=H_u.device).expand(B, M, M)
    P_null = eyeM - H_u @ HH_inv @ HuH                         # (B, M, M)

    A = P_null @ G_T.conj().transpose(-1, -2) @ G_T @ P_null
    A = 0.5 * (A + A.conj().transpose(-1, -2))                 # Hermitian-ise
    _, eigvecs = torch.linalg.eigh(A)                          # ascending
    w_T = eigvecs[..., -1]                                     # (B, M)
    w_T = w_T / (w_T.norm(dim=-1, keepdim=True) + 1e-9)

    # ---- BS receive combiner (MRC matched to G_T w_T) ----
    gw = torch.einsum("bmn,bn->bm", G_T, w_T)
    u_T = gw / (gw.norm(dim=-1, keepdim=True) + 1e-9)

    # ---- Effective scalar gains ----
    def absq(a, b):
        # |<a, b>|^2 over the last dim  (a, b: (B, M) complex)
        return (a.conj() * b).sum(-1).abs() ** 2

    def gscat(u, G, w):
        Gw = torch.einsum("bmn,bn->bm", G, w)
        return (u.conj() * Gw).sum(-1).abs() ** 2

    gff = absq(h_f, w_f)
    gfn = absq(h_f, w_n)
    gfT = absq(h_f, w_T)

    gnn = absq(h_n, w_n)
    gnT = absq(h_n, w_T)
    # gnf, gsf, gsn unused under perfect SIC (delta = 0) but kept for clarity:
    # gnf = absq(h_n, w_f)

    gsT = gscat(u_T, G_T, w_T)

    # ---- SINRs (perfect SIC: delta_f = delta_n = 0) ----
    a_n, a_f, a_T = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    Pn = a_n * P_tot
    Pf = a_f * P_tot
    PT = a_T * P_tot

    snr_f = (gff * Pf) / (gfn * Pn + gfT * PT + sigma2)
    snr_n = (gnn * Pn) / (gnT * PT + sigma2)
    snr_s = (gsT * PT) / sigma2

    R_n = torch.log2(1.0 + snr_n)
    R_f = torch.log2(1.0 + snr_f)
    R_s = torch.log2(1.0 + snr_s)
    return R_n, R_f, R_s, snr_n, snr_f, snr_s


# ============================================================
# 4. Loss
# ============================================================
def loss_fn(R_n, R_f, R_s, R_th_c, R_th_s,
            R_th_n=0.0, R_th_f=0.0,
            w_c=0.7, w_s=0.3,
            lam_c=2.0, lam_s=2.0, lam_n=0.0, lam_f=0.0):
    """
    Lagrangian relaxation of P3 (eq. 23-27).

    Required (paper's P3):
        pen_c = relu(R_th_c - (R_n + R_f))   # eq. 24, sum comm rate
        pen_s = relu(R_th_s - R_s)            # eq. 25, sensing rate

    Optional NOMA-fairness extensions (not in the paper, set lam_n=lam_f=0
    to disable):
        pen_n = relu(R_th_n - R_n)            # near-user floor
        pen_f = relu(R_th_f - R_f)            # far-user floor
    """
    R_sum = w_c * (R_n + R_f) + w_s * R_s
    pen_c = torch.relu(R_th_c - (R_n + R_f))
    pen_s = torch.relu(R_th_s - R_s)
    pen_n = torch.relu(R_th_n - R_n)
    pen_f = torch.relu(R_th_f - R_f)
    L = (-R_sum.mean()
         + lam_c * pen_c.mean() + lam_s * pen_s.mean()
         + lam_n * pen_n.mean() + lam_f * pen_f.mean())
    return L, R_sum, pen_c, pen_s


# ============================================================
# 5. Baseline reference (random RIS + fixed alloc, MATLAB-style)
# ============================================================
@torch.no_grad()
def baseline_eval(loader, device, ds, a_baseline=(0.30, 0.40, 0.30)):
    """Random uniform RIS phases + fixed (a_n, a_f, a_T) for sanity reference."""
    N = ds["N"]
    P_tot, sigma2, beta_T = ds["P_tot"], ds["sigma2"], ds["beta_T"]
    R_th_c, R_th_s = ds["R_th_c"], ds["R_th_s"]
    agg = dict(R=0, qany=0, n=0)
    for batch in loader:
        H_BR, h_RDn, h_RDf, h_RT, h_TR = [b.to(device) for b in batch]
        B = H_BR.size(0)
        phi = 2 * np.pi * torch.rand(B, N, device=device)
        theta = torch.complex(torch.cos(phi), torch.sin(phi)).to(H_BR.dtype)
        alpha = torch.tensor(a_baseline, device=device).expand(B, 3)
        R_n, R_f, R_s, *_ = compute_rates(
            theta, alpha, H_BR, h_RDn, h_RDf, h_RT, h_TR, P_tot, sigma2, beta_T)
        R_sum = 0.7 * (R_n + R_f) + 0.3 * R_s
        viol = (((R_n + R_f) < R_th_c) | (R_s < R_th_s)).float()
        agg["R"]    += R_sum.mean().item() * B
        agg["qany"] += viol.mean().item() * B
        agg["n"]    += B
    return agg["R"] / agg["n"], agg["qany"] / agg["n"]


# ============================================================
# 6. Train / eval loop
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat",    default="ISAC_RIS_NOMA_channels_v3_easy.mat")
    parser.add_argument("--epochs", type=int,   default=80)
    parser.add_argument("--batch",  type=int,   default=256)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--lam_c",  type=float, default=2.0)
    parser.add_argument("--lam_s",  type=float, default=2.0)
    parser.add_argument("--lam_n",  type=float, default=0.0,
                        help="weight on near-user rate floor (NOMA fairness)")
    parser.add_argument("--lam_f",  type=float, default=0.0,
                        help="weight on far-user rate floor (NOMA fairness)")
    parser.add_argument("--R_th_n", type=float, default=1.0,
                        help="near-user rate floor (bits/s/Hz), active if lam_n>0")
    parser.add_argument("--R_th_f", type=float, default=1.0,
                        help="far-user rate floor (bits/s/Hz), active if lam_f>0")
    parser.add_argument("--hidden", type=int,   default=256)
    parser.add_argument("--seed",   type=int,   default=7)
    parser.add_argument("--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out",    default="policy_best.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- data ---
    ds = load_dataset(Path(args.mat))
    N, M = ds["N"], ds["M"]
    P_tot, sigma2, beta_T = ds["P_tot"], ds["sigma2"], ds["beta_T"]
    R_th_c, R_th_s = ds["R_th_c"], ds["R_th_s"]

    tensors = (torch.from_numpy(ds["H_BR"]),
               torch.from_numpy(ds["h_RDn"]),
               torch.from_numpy(ds["h_RDf"]),
               torch.from_numpy(ds["h_RT"]),
               torch.from_numpy(ds["h_TR"]))
    full = TensorDataset(*tensors)
    n_train = int(0.9 * len(full))
    n_val   = len(full) - n_train
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch)

    print(f"Samples: train={n_train} val={n_val} | N={N} M={M}")
    print(f"R_th_c={R_th_c}  R_th_s={R_th_s}  P_tot={P_tot:.3e}")

    R_base, q_base = baseline_eval(val_loader, device, ds)
    print(f"[Baseline] random-RIS + (0.30/0.40/0.30):  "
          f"R_DL_sum={R_base:.4f}  qos_viol_any={q_base:.4f}\n")

    # --- model ---
    net = PolicyNet(N, M, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def run_epoch(loader, train_mode):
        net.train(train_mode)
        agg = dict(L=0, R=0, qc=0, qs=0, qany=0, n=0)
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for batch in loader:
                H_BR, h_RDn, h_RDf, h_RT, h_TR = [b.to(device) for b in batch]
                feats = channels_to_features(H_BR, h_RDn, h_RDf, h_RT, h_TR)
                theta, alpha = net(feats)
                theta = theta.to(H_BR.dtype)        # cast to complex64 for physics
                R_n, R_f, R_s, *_ = compute_rates(
                    theta, alpha, H_BR, h_RDn, h_RDf, h_RT, h_TR,
                    P_tot, sigma2, beta_T)
                L, R_sum, _, _ = loss_fn(
                    R_n, R_f, R_s, R_th_c, R_th_s,
                    R_th_n=args.R_th_n, R_th_f=args.R_th_f,
                    lam_c=args.lam_c, lam_s=args.lam_s,
                    lam_n=args.lam_n, lam_f=args.lam_f)
                if train_mode:
                    opt.zero_grad(); L.backward(); opt.step()

                B = H_BR.size(0)
                viol_c   = ((R_n + R_f) < R_th_c).float()
                viol_s   = (R_s < R_th_s).float()
                viol_any = ((viol_c + viol_s) > 0).float()
                agg["L"]    += L.item()    * B
                agg["R"]    += R_sum.mean().item() * B
                agg["qc"]   += viol_c.mean().item() * B
                agg["qs"]   += viol_s.mean().item() * B
                agg["qany"] += viol_any.mean().item() * B
                agg["n"]    += B
        return {k: v / agg["n"] for k, v in agg.items() if k != "n"}

    print(f"{'epoch':>5} | {'L_tr':>7} {'R_tr':>6} {'qany_tr':>7} | "
          f"{'L_val':>7} {'R_val':>6} {'qc_v':>5} {'qs_v':>5} {'qany_v':>6}")

    best = -float("inf")
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        print(f"{ep:5d} | "
              f"{tr['L']:7.3f} {tr['R']:6.3f} {tr['qany']:7.3f} | "
              f"{va['L']:7.3f} {va['R']:6.3f} "
              f"{va['qc']:5.3f} {va['qs']:5.3f} {va['qany']:6.3f}")

        # Score: prefer feasible-then-rate. Penalise infeasibility heavily.
        score = va["R"] - 5.0 * va["qany"]
        if score > best:
            best = score
            torch.save({
                "state_dict": net.state_dict(),
                "args": vars(args),
                "val": va,
                "baseline": {"R": R_base, "qany": q_base},
            }, args.out)

    print(f"\nBest checkpoint score (R - 5*qany) = {best:.4f}  ->  {args.out}")
    print(f"Baseline reference        : R={R_base:.4f}  qany={q_base:.4f}")


if __name__ == "__main__":
    main()

"""
gen_fig6_channels.py
====================================================================
Generates two held-out test-channel sets needed for the six-scheme
NOMA comparison (results/figures/fig6a_six_schemes.png):

  1. RIS test channels  (N=8, M=4) -- same physical scenario as
     matlab/miso_isac_noma_v3_fair_chatpgt.m (distances, Nakagami-m
     shape parameters, path-loss exponents), with a fresh RNG seed
     so this is an independent held-out test set, not the training set.

  2. No-RIS (direct-link) test channels (M=4) -- the M=4 "fair"
     no-RIS scenario used to train policy_noris_best.pt: same
     distances/Nakagami-m shapes as the RIS branch's near/far/target
     links, path-loss exponents (alpha_BDn=3.8, alpha_BDf=3.5,
     alpha_BT=4.36), and a 29 dB partial-blockage factor applied to
     the two user links (not the target link). Fresh RNG seed.

Channel amplitudes/phases only depend on distance, Nakagami-m shape,
and path-loss exponent -- not on P_tot or beta_T (those enter later,
at the SINR/rate stage). Generating held-out test channels here and
evaluating them against the already-trained policy checkpoints (whose
.pt files embed the exact P_tot / beta_T / sigma2 / R_th_c / R_th_s
they were trained with) is physically consistent with the trained
models.

Output (NumPy .npz, consumed by fig6_six_scheme_compare.py):
  fig6_ris_test_channels.npz    H_BR, h_RDn, h_RDf, h_RT, h_TR
  fig6_noris_test_channels.npz  h_BDn, h_BDf, h_BT
"""
import numpy as np
import os

# Written alongside this script in src/; fig6_six_scheme_compare.py
# reads from the same location.
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------
# Nakagami-m fading channel generator (ports nakagami_channel.m)
# --------------------------------------------------------------
def nakagami_channel(rows, cols, m_val, path_loss, rng):
    m_int = int(round(m_val))
    expo = -np.log(rng.random((rows, cols, m_int)))
    amp_sq = expo.sum(axis=2) / m_val
    amp = np.sqrt(amp_sq)
    phase = 2 * np.pi * rng.random((rows, cols))
    H = np.sqrt(path_loss) * amp * np.exp(1j * phase)
    return H


def gen_ris_test_set(num_samples=5000, seed=12345):
    """Mirrors miso_isac_noma_v3_fair_chatpgt.m channel statistics."""
    rng = np.random.default_rng(seed)

    M, N = 4, 8

    # Nakagami-m shape parameters (identical to the .m file)
    m_BR, m_RDn, m_RDf, m_RT, m_TR = 2, 1, 1, 3, 3

    # Distances (identical to the .m file)
    d_BR, d_RDn, d_RDf, d_RT, d_TR = 8, 3, 25, 6, 6

    # Path loss (identical to the .m file)
    PL0 = 10 ** (-3.0)
    alpha_BR, alpha_RDn, alpha_RDf, alpha_RT, alpha_TR = 2.2, 2.8, 2.8, 2.3, 2.3
    PL_BR = PL0 * d_BR ** (-alpha_BR)
    PL_RDn = PL0 * d_RDn ** (-alpha_RDn)
    PL_RDf = PL0 * d_RDf ** (-alpha_RDf)
    PL_RT = PL0 * d_RT ** (-alpha_RT)
    PL_TR = PL0 * d_TR ** (-alpha_TR)

    H_BR_all = np.zeros((num_samples, N, M), dtype=np.complex64)
    h_RDn_all = np.zeros((num_samples, N), dtype=np.complex64)
    h_RDf_all = np.zeros((num_samples, N), dtype=np.complex64)
    h_RT_all = np.zeros((num_samples, N), dtype=np.complex64)
    h_TR_all = np.zeros((num_samples, N), dtype=np.complex64)

    for s in range(num_samples):
        H_BR_all[s] = nakagami_channel(N, M, m_BR, PL_BR, rng)
        h_RDn_all[s] = nakagami_channel(N, 1, m_RDn, PL_RDn, rng)[:, 0]
        h_RDf_all[s] = nakagami_channel(N, 1, m_RDf, PL_RDf, rng)[:, 0]
        h_RT_all[s] = nakagami_channel(N, 1, m_RT, PL_RT, rng)[:, 0]
        h_TR_all[s] = nakagami_channel(N, 1, m_TR, PL_TR, rng)[:, 0]

    np.savez(f"{OUT_DIR}/fig6_ris_test_channels.npz",
             H_BR=H_BR_all, h_RDn=h_RDn_all, h_RDf=h_RDf_all,
             h_RT=h_RT_all, h_TR=h_TR_all, N=N, M=M,
             num_samples=num_samples, seed=seed)
    print(f"[RIS test set]    saved {num_samples} samples (N={N}, M={M}) "
          f"-> fig6_ris_test_channels.npz")


def gen_noris_test_set(num_samples=5000, seed=12346):
    """M=4 "fair" no-RIS scenario matching the policy_noris_best.pt
    training data: same near/far/target distances and Nakagami-m
    shapes as the RIS branch's direct links, plus a 29 dB partial
    blockage factor on the two user links."""
    rng = np.random.default_rng(seed)

    M = 4
    m_BDn, m_BDf, m_BT = 1, 1, 3
    d_BDn, d_BDf, d_BT = 10, 20, 25
    alpha_BDn, alpha_BDf, alpha_BT = 3.8, 3.5, 4.36

    PL0 = 10 ** (-3.0)
    PL_BDn = PL0 * d_BDn ** (-alpha_BDn)
    PL_BDf = PL0 * d_BDf ** (-alpha_BDf)
    PL_BT = PL0 * d_BT ** (-alpha_BT)

    # Partial-blockage factor applied to the two user links only (not
    # the target link).
    block_dB = 29
    PL_BDn = PL_BDn * 10 ** (-block_dB / 10)
    PL_BDf = PL_BDf * 10 ** (-block_dB / 10)

    h_BDn_all = np.zeros((num_samples, M), dtype=np.complex64)
    h_BDf_all = np.zeros((num_samples, M), dtype=np.complex64)
    h_BT_all = np.zeros((num_samples, M), dtype=np.complex64)

    for s in range(num_samples):
        h_BDn_all[s] = nakagami_channel(M, 1, m_BDn, PL_BDn, rng)[:, 0]
        h_BDf_all[s] = nakagami_channel(M, 1, m_BDf, PL_BDf, rng)[:, 0]
        h_BT_all[s] = nakagami_channel(M, 1, m_BT, PL_BT, rng)[:, 0]

    np.savez(f"{OUT_DIR}/fig6_noris_test_channels.npz",
             h_BDn=h_BDn_all, h_BDf=h_BDf_all, h_BT=h_BT_all,
             M=M, num_samples=num_samples, seed=seed)
    print(f"[No-RIS test set] saved {num_samples} samples (M={M}) "
          f"-> fig6_noris_test_channels.npz")


if __name__ == "__main__":
    gen_ris_test_set()
    gen_noris_test_set()

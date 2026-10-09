%% =========================================================
%  gen_fig6_noris_test_channels.m
%  ---------------------------------------------------------
%  Held-out no-RIS (direct-link) test-channel generator for the
%  "NOMA + No-RIS" curve in the six-scheme comparison. M=4 "fair"
%  no-RIS scenario matching the policy_noris_best.pt training data:
%  same near/far/target distances and Nakagami-m shapes as the RIS
%  branch's direct links, plus a 29 dB partial-blockage factor applied
%  to the two user links (not the target link). Fresh RNG seed so this
%  stays an independent held-out set, not the training data.
%
%  Output: ISAC_NOMA_channels_fig6_noris_TEST.mat
%    h_BDn_all (M,1,num_samples)   BS->near-user direct channel
%    h_BDf_all (M,1,num_samples)   BS->far-user direct channel
%    h_BT_all  (M,1,num_samples)   BS->target direct channel
%    M, num_samples
% =========================================================

clc; clear; close all;
rng(12346, 'twister');   % fresh seed, distinct from the RIS test set

%% --- System dimensions ---
M = 4;
num_samples = 5000;

%% --- Nakagami-m shape parameters ---
m_BDn = 1; m_BDf = 1; m_BT = 3;

%% --- Distances (m) ---
d_BDn = 10; d_BDf = 20; d_BT = 25;

%% --- Path-loss exponents ---
alpha_BDn = 3.8; alpha_BDf = 3.5; alpha_BT = 4.36;

PL0    = 10^(-3.0);
PL_BDn = PL0 * d_BDn^(-alpha_BDn);
PL_BDf = PL0 * d_BDf^(-alpha_BDf);
PL_BT  = PL0 * d_BT ^(-alpha_BT);

%% --- Partial-blockage factor (user links only, NOT the target link) ---
block_dB = 29;
PL_BDn = PL_BDn * 10^(-block_dB/10);
PL_BDf = PL_BDf * 10^(-block_dB/10);

fprintf('=== No-RIS test-channel generator (seed=12346, num_samples=%d) ===\n', num_samples);

%% --- Storage ---
h_BDn_all = zeros(M, 1, num_samples);
h_BDf_all = zeros(M, 1, num_samples);
h_BT_all  = zeros(M, 1, num_samples);

for s = 1:num_samples
    h_BDn_all(:,:,s) = nakagami_channel(M, 1, m_BDn, PL_BDn);
    h_BDf_all(:,:,s) = nakagami_channel(M, 1, m_BDf, PL_BDf);
    h_BT_all(:,:,s)  = nakagami_channel(M, 1, m_BT,  PL_BT);
end

output_mat = fullfile(pwd, 'ISAC_NOMA_channels_fig6_noris_TEST.mat');
save(output_mat, 'h_BDn_all','h_BDf_all','h_BT_all','M','num_samples','-v7.3');
fprintf('Saved -> %s\n', output_mat);

%% ---------------- LOCAL FUNCTION ----------------
function H = nakagami_channel(rows, cols, m_val, path_loss)
    m_int = round(m_val);
    assert(abs(m_val - m_int) < 1e-12 && m_int >= 1, 'm must be positive integer.');
    expo   = -log(rand(rows, cols, m_int));
    amp_sq = sum(expo, 3) / m_val;
    amp    = sqrt(amp_sq);
    phase  = 2*pi * rand(rows, cols);
    H      = sqrt(path_loss) * amp .* exp(1j*phase);
end

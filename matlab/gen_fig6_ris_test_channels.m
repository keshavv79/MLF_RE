%% =========================================================
%  gen_fig6_ris_test_channels.m
%  ---------------------------------------------------------
%  Held-out RIS test-channel generator for the six-scheme NOMA
%  comparison. Same physical scenario (distances, Nakagami-m shape
%  parameters, path-loss exponents) as miso_isac_noma_v3_fair_chatpgt.m,
%  with a fresh RNG seed so this is an independent held-out set, not
%  the training set.
%
%  Channel amplitude/phase statistics only depend on distance,
%  Nakagami-m shape, and path-loss exponent -- not on P_tot or beta_T
%  (those enter later, at the SINR/rate stage) -- so this file is
%  self-contained and does not need beta_T/P_tot/power-split constants.
%
%  Output: ISAC_RIS_NOMA_channels_fig6_TEST.mat
%    H_BR_all  (N,M,num_samples)   BS->RIS channel
%    h_RDn_all (N,1,num_samples)   RIS->near-user channel
%    h_RDf_all (N,1,num_samples)   RIS->far-user channel
%    h_RT_all  (N,1,num_samples)   RIS->target channel
%    h_TR_all  (N,1,num_samples)   target->RIS channel
%    N, M, num_samples
% =========================================================

clc; clear; close all;
rng(12345, 'twister');   % fresh seed -> independent test set

%% --- System dimensions (match miso_isac_noma_v3_fair_chatpgt.m) ---
M = 4;
N = 8;
num_samples = 5000;

%% --- Nakagami-m shape parameters ---
m_BR=2; m_RDn=1; m_RDf=1; m_RT=3; m_TR=3;

%% --- Distances (m) ---
d_BR=8; d_RDn=3; d_RDf=25; d_RT=6; d_TR=6;

%% --- Path loss ---
PL0 = 10^(-3.0);
alpha_BR=2.2; alpha_RDn=2.8; alpha_RDf=2.8; alpha_RT=2.3; alpha_TR=2.3;

PL_BR  = PL0*d_BR^(-alpha_BR);
PL_RDn = PL0*d_RDn^(-alpha_RDn);
PL_RDf = PL0*d_RDf^(-alpha_RDf);
PL_RT  = PL0*d_RT^(-alpha_RT);
PL_TR  = PL0*d_TR^(-alpha_TR);

fprintf('=== RIS test-channel generator (seed=12345, num_samples=%d) ===\n', num_samples);

%% --- Storage ---
H_BR_all  = zeros(N, M, num_samples);
h_RDn_all = zeros(N, 1, num_samples);
h_RDf_all = zeros(N, 1, num_samples);
h_RT_all  = zeros(N, 1, num_samples);
h_TR_all  = zeros(N, 1, num_samples);

for s = 1:num_samples
    H_BR_all(:,:,s)  = nakagami_channel(N, M, m_BR,  PL_BR);
    h_RDn_all(:,:,s) = nakagami_channel(N, 1, m_RDn, PL_RDn);
    h_RDf_all(:,:,s) = nakagami_channel(N, 1, m_RDf, PL_RDf);
    h_RT_all(:,:,s)  = nakagami_channel(N, 1, m_RT,  PL_RT);
    h_TR_all(:,:,s)  = nakagami_channel(N, 1, m_TR,  PL_TR);
end

output_mat = fullfile(pwd, 'ISAC_RIS_NOMA_channels_fig6_TEST.mat');
save(output_mat, 'H_BR_all','h_RDn_all','h_RDf_all','h_RT_all','h_TR_all', ...
     'N','M','num_samples','-v7.3');
fprintf('Saved -> %s\n', output_mat);

%% ---------------- LOCAL FUNCTION ----------------
function H = nakagami_channel(rows, cols, m_val, path_loss)
    m_int = round(m_val);
    expo = -log(rand(rows,cols,m_int));
    amp_sq = sum(expo,3)/m_val;
    amp = sqrt(amp_sq);
    phase = 2*pi*rand(rows,cols);
    H = sqrt(path_loss) * amp .* exp(1j*phase);
end

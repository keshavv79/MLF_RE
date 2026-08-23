%% =========================================================
%  Test-set generator for the OMA-TDMA v3-easy DL policy.
%  Identical physics to miso_isac_oma_v3_easy.m but:
%    - different RNG seed (42 vs 7)  -> disjoint channel realisations
%    - smaller sample count (5000)   -> evaluation-only set
%    - writes ISAC_RIS_OMA_channels_v3_easy_TEST.mat
%  Only the raw channels + scalars are needed by the Python evaluator;
%  OMA baseline SINR/rate fields are still computed so MATLAB-side numbers
%  print for sanity comparison.
% =========================================================

clc; clear; close all;
rng(42, 'twister');               % <-- different seed from training set

%% --- System dimensions (must match training) ---
M           = 2;
N           = 8;
num_samples = 5000;               % test-set size

%% --- Nakagami-m shape parameters ---
m_BR  = 2; m_RDn = 1; m_RDf = 1; m_RT  = 3; m_TR  = 3;
m_BDn = 1; m_BDf = 1;

%% --- Distances (m) -- same as OMA training set ---
d_BR=8; d_RDn=5; d_RDf=15; d_RT=6; d_TR=6; d_BDn=10; d_BDf=25;

%% --- Path loss ---
PL0       = 10^(-3.0);
alpha_BR  = 2.2; alpha_RDn=2.8; alpha_RDf=2.8;
alpha_RT  = 2.3; alpha_TR =2.3; alpha_BDn=3.8; alpha_BDf=3.8;

PL_BR  = PL0 * d_BR ^(-alpha_BR);
PL_RDn = PL0 * d_RDn^(-alpha_RDn);
PL_RDf = PL0 * d_RDf^(-alpha_RDf);
PL_RT  = PL0 * d_RT ^(-alpha_RT);
PL_TR  = PL0 * d_TR ^(-alpha_TR);
PL_BDn = PL0 * d_BDn^(-alpha_BDn);
PL_BDf = PL0 * d_BDf^(-alpha_BDf);

beta_T_dB = 40; beta_T = 10^(beta_T_dB/10);

%% --- Power & noise ---
P_tot_dBm = 40;
P_tot     = 10^((P_tot_dBm-30)/10);
BW=1e6; T_noise=290; k_B=1.38e-23; NF_dB=6;
sigma2 = k_B * T_noise * BW * 10^(NF_dB/10);

%% --- OMA time split + thresholds (must match training set) ---
t_n = 1/3; t_f = 1/3; t_T = 1/3;
eta_ris        = 1.0;
include_direct = false;

R_th_c = 11.0;                    % <-- matches miso_isac_oma_v3_easy.m
R_th_s = 1.0;

fprintf('=== OMA TEST-SET GENERATOR (seed=42, num_samples=%d) ===\n', num_samples);

%% --- Storage ---
H_BR_all   = zeros(N, M, num_samples);
h_RDn_all  = zeros(N, 1, num_samples);
h_RDf_all  = zeros(N, 1, num_samples);
h_RT_all   = zeros(N, 1, num_samples);
h_TR_all   = zeros(N, 1, num_samples);
h_BDn_all  = zeros(M, 1, num_samples);
h_BDf_all  = zeros(M, 1, num_samples);
Theta_all  = zeros(N, N, num_samples);

R_f_all = zeros(1, num_samples);
R_n_all = zeros(1, num_samples);
R_s_all = zeros(1, num_samples);
qos_viol_any = false(1, num_samples);

%% ===================== MONTE CARLO =====================
for s = 1:num_samples
    H_BR  = nakagami_channel(N, M, m_BR,  PL_BR);
    h_RDn = nakagami_channel(N, 1, m_RDn, PL_RDn);
    h_RDf = nakagami_channel(N, 1, m_RDf, PL_RDf);
    h_RT  = nakagami_channel(N, 1, m_RT,  PL_RT);
    h_TR  = nakagami_channel(N, 1, m_TR,  PL_TR);

    if include_direct
        h_BDn = nakagami_channel(M, 1, m_BDn, PL_BDn);
        h_BDf = nakagami_channel(M, 1, m_BDf, PL_BDf);
    else
        h_BDn = zeros(M,1); h_BDf = zeros(M,1);
    end

    % Random continuous RIS (just so the .mat has a Theta field; Python
    % evaluator overrides this with the policy's prediction)
    theta_raw = 2*pi * rand(N, 1);
    Theta     = eta_ris * diag(exp(1j * theta_raw));

    % MATLAB-side OMA baseline rates (random RIS, per-user MRT, equal time)
    h_n = (h_RDn.' * Theta * H_BR).' + h_BDn;
    h_f = (h_RDf.' * Theta * H_BR).' + h_BDf;
    G_T = sqrt(beta_T) * (H_BR' * Theta * h_TR) * (h_RT.' * Theta * H_BR);

    w_n = h_n / max(norm(h_n), 1e-15);
    w_f = h_f / max(norm(h_f), 1e-15);

    A_s = G_T' * G_T; A_s = (A_s + A_s')/2;
    [V_A, D_A] = eig(A_s);
    [~, idx_max] = max(real(diag(D_A)));
    w_T = V_A(:, idx_max);
    w_T = w_T / max(norm(w_T), 1e-15);

    g_T_eff = G_T * w_T;
    if norm(g_T_eff) > 1e-15
        u_T = g_T_eff / norm(g_T_eff);
    else
        u_T = zeros(M,1); u_T(1) = 1;
    end

    g_nn = abs(h_n' * w_n)^2;
    g_ff = abs(h_f' * w_f)^2;
    g_sT = abs(u_T' * G_T * w_T)^2;

    snr_n = g_nn * P_tot / sigma2;
    snr_f = g_ff * P_tot / sigma2;
    snr_s = g_sT * P_tot / sigma2;

    R_n_all(s) = t_n * log2(1 + snr_n);
    R_f_all(s) = t_f * log2(1 + snr_f);
    R_s_all(s) = t_T * log2(1 + snr_s);

    qos_viol_any(s) = (R_n_all(s) + R_f_all(s) < R_th_c) || (R_s_all(s) < R_th_s);

    H_BR_all (:,:,s) = H_BR;
    h_RDn_all(:,:,s) = h_RDn;
    h_RDf_all(:,:,s) = h_RDf;
    h_RT_all (:,:,s) = h_RT;
    h_TR_all (:,:,s) = h_TR;
    h_BDn_all(:,:,s) = h_BDn;
    h_BDf_all(:,:,s) = h_BDf;
    Theta_all(:,:,s) = Theta;
end

w_c_obj = 0.7; w_s_obj = 0.3;
R_DL_sum_avg = mean(w_c_obj * (R_n_all + R_f_all) + w_s_obj * R_s_all);

fprintf('MATLAB OMA baseline on this test set (random RIS, MRT, equal time):\n');
fprintf('  R_DL_sum = %.4f  bits/s/Hz\n', R_DL_sum_avg);
fprintf('  qos_viol = %.4f\n', mean(qos_viol_any));

%% --- Save (same field layout load_dataset() expects) ---
output_mat_file = fullfile(pwd, 'ISAC_RIS_OMA_channels_v3_easy_TEST.mat');
save(output_mat_file, ...
    'H_BR_all','h_RDn_all','h_RDf_all','h_RT_all','h_TR_all', ...
    'h_BDn_all','h_BDf_all','Theta_all', ...
    'R_n_all','R_f_all','R_s_all','qos_viol_any', ...
    'N','M','num_samples', ...
    't_n','t_f','t_T','P_tot','sigma2', ...
    'beta_T','eta_ris','include_direct', ...
    'R_th_c','R_th_s','-v7.3');

fprintf('Test set saved -> %s\n', output_mat_file);

%% --- Nakagami helper ---
function H = nakagami_channel(rows, cols, m_val, path_loss)
    m_int = round(m_val);
    assert(abs(m_val - m_int) < 1e-12 && m_int >= 1, 'm must be positive integer.');
    expo   = -log(rand(rows, cols, m_int));
    amp_sq = sum(expo, 3) / m_val;
    amp    = sqrt(amp_sq);
    phase  = 2*pi * rand(rows, cols);
    H      = sqrt(path_loss) * amp .* exp(1j*phase);
end

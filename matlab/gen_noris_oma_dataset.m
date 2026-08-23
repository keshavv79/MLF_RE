%% =========================================================
%  No-RIS MISO ISAC-OMA dataset generator
%  ---------------------------------------------------------
%  OMA-TDMA counterpart of gen_noris_dataset.m (NOMA).
%  Same direct-link physical scenario (no RIS) -> identical channels:
%      h_BDn : BS -> Near User   (M x 1)
%      h_BDf : BS -> Far  User   (M x 1)
%      h_BT  : BS -> Target      (M x 1)
%  Sensing channel: monostatic round-trip  G_T = sqrt(beta_T) * h_BT * h_BT.'
%
%  The ONLY difference from the NOMA no-RIS file is the access scheme:
%    OMA -> 3 orthogonal time slots (t_n,t_f,t_T), per-user MRT,
%           unconstrained sensing BF, full P_tot per slot, no interference.
%    Rates: R_k = t_k * log2(1 + g_kk * P_tot / sigma2).
%
%  Thresholds match miso_isac_oma_v3_easy.m (R_th_c=11.0, R_th_s=1.0) so the
%  no-RIS branch is QoS-consistent with the OMA-RIS branch in fig8 / eval.
%
%  Output: ISAC_OMA_channels_noris.mat
%          (place under OMA_new/no_ris/ for the fig8_oma sweep)
% =========================================================

clc; clear; close all;
rng(7, 'twister');

%% --- System dimensions ---
M           = 2;
num_samples = 10000;

%% --- Nakagami-m shape parameters (same as NOMA no-RIS) ---
m_BDn = 1;
m_BDf = 1;
m_BT  = 3;          % LOS-dominated direct target return

%% --- Distances (m) -- same direct-link scenario as NOMA no-RIS ---
d_BDn = 10;
d_BDf = 20;
d_BT  = 25;

%% --- Path-loss exponents (same as NOMA no-RIS) ---
alpha_BDn = 3.8;
alpha_BDf = 3.5;
alpha_BT  = 4.5;

PL0    = 10^(-3.0);
PL_BDn = PL0 * d_BDn^(-alpha_BDn);
PL_BDf = PL0 * d_BDf^(-alpha_BDf);
PL_BT  = PL0 * d_BT ^(-alpha_BT);

beta_T_dB = 40;
beta_T    = 10^(beta_T_dB/10);

%% --- Power & noise (matches v3_easy: 40 dBm, 1 MHz BW, 6 dB NF) ---
P_tot_dBm = 40;
P_tot     = 10^((P_tot_dBm-30)/10);

BW      = 1e6;
T_noise = 290;
k_B     = 1.38e-23;
NF_dB   = 6;
sigma2  = k_B * T_noise * BW * 10^(NF_dB/10);

%% --- OMA time split (DL trainer will override) ---
t_n = 1/3;  t_f = 1/3;  t_T = 1/3;
assert(abs(t_n + t_f + t_T - 1) < 1e-9);

%% --- QoS thresholds (match miso_isac_oma_v3_easy.m) ---
R_th_n_out = 2.0;
R_th_f_out = 0.8;
R_th_s_out = 0.3;
R_th_c     = 11.0;
R_th_s     = 1.0;

%% --- Sanity print ---
fprintf('========================================================\n');
fprintf(' No-RIS MISO ISAC-OMA dataset  (M=%d, samples=%d)\n', M, num_samples);
fprintf('========================================================\n');
fprintf('Path loss (dB):\n');
fprintf('  BS->Dn = %.2f  BS->Df = %.2f  BS->T = %.2f\n', ...
        10*log10(PL_BDn), 10*log10(PL_BDf), 10*log10(PL_BT));
fprintf('Power:  P_tot=%d dBm  sigma^2=%.2f dBm  SNR budget=%.2f dB\n', ...
        P_tot_dBm, 10*log10(sigma2)+30, 10*log10(P_tot/sigma2));
fprintf('OMA baseline: t_n=%.3f t_f=%.3f t_T=%.3f  (equal time)\n\n', t_n, t_f, t_T);

%% --- Storage ---
h_BDn_all = zeros(M, 1, num_samples);
h_BDf_all = zeros(M, 1, num_samples);
h_BT_all  = zeros(M, 1, num_samples);

w_n_all = zeros(M, 1, num_samples);
w_f_all = zeros(M, 1, num_samples);
w_T_all = zeros(M, 1, num_samples);
u_T_all = zeros(M, 1, num_samples);

g_nn_all = zeros(1, num_samples);
g_ff_all = zeros(1, num_samples);
g_sT_all = zeros(1, num_samples);

SINR_f_all = zeros(1, num_samples);
SINR_n_all = zeros(1, num_samples);
SINR_s_all = zeros(1, num_samples);
R_f_all    = zeros(1, num_samples);
R_n_all    = zeros(1, num_samples);
R_s_all    = zeros(1, num_samples);

qos_viol_c_all = false(1, num_samples);
qos_viol_s_all = false(1, num_samples);
qos_viol_any   = false(1, num_samples);

%% ===================== MAIN MONTE CARLO LOOP =====================
for s = 1:num_samples

    h_n  = nakagami_channel(M, 1, m_BDn, PL_BDn);   % BS->Dn
    h_f  = nakagami_channel(M, 1, m_BDf, PL_BDf);   % BS->Df
    h_BT = nakagami_channel(M, 1, m_BT,  PL_BT );   % BS->T

    % Monostatic round-trip sensing channel (M x M, rank-1)
    G_T = sqrt(beta_T) * (h_BT * h_BT.');

    % OMA beamforming: per-user MRT + unconstrained sensing BF
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
        u_T = zeros(M, 1); u_T(1) = 1;
    end

    % Effective gains (no cross terms in OMA)
    g_nn = abs(h_n' * w_n)^2;
    g_ff = abs(h_f' * w_f)^2;
    g_sT = abs(u_T' * G_T * w_T)^2;

    % SINR (full power per slot, no interference)
    snr_n = g_nn * P_tot / sigma2;
    snr_f = g_ff * P_tot / sigma2;
    snr_s = g_sT * P_tot / sigma2;

    % Time-weighted rates
    R_n = t_n * log2(1 + snr_n);
    R_f = t_f * log2(1 + snr_f);
    R_s = t_T * log2(1 + snr_s);

    h_BDn_all(:,:,s) = h_n;
    h_BDf_all(:,:,s) = h_f;
    h_BT_all (:,:,s) = h_BT;

    w_n_all(:,:,s) = w_n;
    w_f_all(:,:,s) = w_f;
    w_T_all(:,:,s) = w_T;
    u_T_all(:,:,s) = u_T;

    g_nn_all(s) = g_nn;
    g_ff_all(s) = g_ff;
    g_sT_all(s) = g_sT;

    SINR_n_all(s) = snr_n;
    SINR_f_all(s) = snr_f;
    SINR_s_all(s) = snr_s;
    R_n_all(s)    = R_n;
    R_f_all(s)    = R_f;
    R_s_all(s)    = R_s;

    R_comm_sum        = R_n + R_f;
    qos_viol_c_all(s) = (R_comm_sum < R_th_c);
    qos_viol_s_all(s) = (R_s        < R_th_s);
    qos_viol_any(s)   = qos_viol_c_all(s) || qos_viol_s_all(s);
end

%% ===================== AGGREGATE RESULTS =====================
w_c_obj = 0.7;  w_s_obj = 0.3;
R_DL_sum_avg = mean(w_c_obj * (R_n_all + R_f_all) + w_s_obj * R_s_all);

fprintf('=== Mean SINR (dB) ===\n');
fprintf('  Near User : %.2f\n', 10*log10(mean(SINR_n_all)+1e-12));
fprintf('  Far  User : %.2f\n', 10*log10(mean(SINR_f_all)+1e-12));
fprintf('  Sensing   : %.2f\n', 10*log10(mean(SINR_s_all)+1e-12));

fprintf('\n=== Average rates (bits/s/Hz, equal-time baseline) ===\n');
fprintf('  R_n = %.4f   R_f = %.4f   R_s = %.4f\n', ...
        mean(R_n_all), mean(R_f_all), mean(R_s_all));
fprintf('  R_DL_sum = %.4f (w_c=%.2f w_s=%.2f)\n\n', ...
        R_DL_sum_avg, w_c_obj, w_s_obj);

P_qos_any = mean(qos_viol_any);
fprintf('Baseline R_DL_sum = %.4f bits/s/Hz, qos_viol = %.1f%%.\n\n', ...
        R_DL_sum_avg, 100*P_qos_any);

%% ===================== SAVE DATASET =====================
output_mat_file = fullfile(pwd, 'ISAC_OMA_channels_noris.mat');
save(output_mat_file, ...
    'h_BDn_all','h_BDf_all','h_BT_all', ...
    'w_n_all','w_f_all','w_T_all','u_T_all', ...
    'g_nn_all','g_ff_all','g_sT_all', ...
    'SINR_f_all','SINR_n_all','SINR_s_all', ...
    'R_f_all','R_n_all','R_s_all', ...
    'qos_viol_c_all','qos_viol_s_all','qos_viol_any', ...
    'R_th_c','R_th_s','R_th_n_out','R_th_f_out','R_th_s_out', ...
    't_n','t_f','t_T', ...
    'M','num_samples','P_tot','sigma2', ...
    'PL_BDn','PL_BDf','PL_BT','beta_T','NF_dB', ...
    '-v7.3');

fprintf('Dataset saved -> %s\n', output_mat_file);

%% ===================== NESTED FUNCTIONS =====================
function H = nakagami_channel(rows, cols, m_val, path_loss)
    m_int = round(m_val);
    assert(abs(m_val - m_int) < 1e-12 && m_int >= 1, 'm must be positive integer.');
    expo   = -log(rand(rows, cols, m_int));
    amp_sq = sum(expo, 3) / m_val;
    amp    = sqrt(amp_sq);
    phase  = 2*pi * rand(rows, cols);
    H      = sqrt(path_loss) * amp .* exp(1j*phase);
end

%% =========================================================
%  MISO RIS-Enabled Downlink ISAC-NOMA System  (v3-EASY)
%  ---------------------------------------------------------
%  NOMA counterpart to miso_isac_oma_v3_easy.m.
%  Identical channel model, distances, path-loss, power, and noise.
%  The ONLY differences vs the OMA file are in the multiple-access scheme:
%
%  NOMA : one shared time slot, cluster MRT (w_c = h_f/||h_f||),
%          power-domain multiplexing (a_n, a_f, a_T), SIC at near user.
%          Null-space projection applied to sensing BF.
%
%  OMA  : three orthogonal time slots (t_n, t_f, t_T),
%          per-user MRT, unconstrained sensing BF, full P_tot per slot.
%
%  RATES (NOMA):
%    SINR_f = a_f*P*gff / (a_n*P*gff + a_T*P*gfT + sigma2)
%    SINR_n = a_n*P*gnn / (a_T*P*gnT + sigma2)          [after SIC]
%    SINR_s = a_T*P*gsT / sigma2                         [after SIC]
%    R_k = log2(1 + SINR_k)
%
%  Output : ISAC_RIS_NOMA_channels_v3_easy.mat
% =========================================================

clc; clear; close all;
rng(7, 'twister');

%% --- System dimensions ---
M           = 2;
N           = 8;
num_samples = 10000;

%% --- Nakagami-m shape parameters ---
m_BR  = 2;
m_RDn = 1;
m_RDf = 1;
m_RT  = 3;
m_TR  = 3;
m_BDn = 1;
m_BDf = 1;

%% --- Distances (m) -- indoor scenario ---
d_BR  = 8;
d_RDn = 5;
d_RDf = 15;
d_RT  = 6;
d_TR  = 6;
d_BDn = 10;
d_BDf = 25;

%% --- Path loss ---
PL0       = 10^(-3.0);
alpha_BR  = 2.2;
alpha_RDn = 2.8;
alpha_RDf = 2.8;
alpha_RT  = 2.3;
alpha_TR  = 2.3;
alpha_BDn = 3.8;
alpha_BDf = 3.8;

PL_BR  = PL0 * d_BR ^(-alpha_BR);
PL_RDn = PL0 * d_RDn^(-alpha_RDn);
PL_RDf = PL0 * d_RDf^(-alpha_RDf);
PL_RT  = PL0 * d_RT ^(-alpha_RT);
PL_TR  = PL0 * d_TR ^(-alpha_TR);
PL_BDn = PL0 * d_BDn^(-alpha_BDn);
PL_BDf = PL0 * d_BDf^(-alpha_BDf);

beta_T_dB = 40;
beta_T    = 10^(beta_T_dB/10);

%% --- Power & noise ---
P_tot_dBm = 40;
P_tot     = 10^((P_tot_dBm-30)/10);

BW      = 1e6;
T_noise = 290;
k_B     = 1.38e-23;
NF_dB   = 6;
sigma2  = k_B * T_noise * BW * 10^(NF_dB/10);

%% --- NOMA power allocation ---
a_f = 0.40;
a_n = 0.30;
a_T = 0.30;
assert(abs(a_n + a_f + a_T - 1) < 1e-9);

%% --- Perfect SIC ---
delta_f = 0.0;
delta_n = 0.0;

%% --- RIS settings (continuous phases) ---
eta_ris        = 1.0;
include_direct = false;

%% --- Outage / QoS thresholds ---
gamma_c_th   = 0.3;
gamma_s_th   = 0.1;
gamma_SIC_th = 0.3;

R_th_n_out = 2.0;
R_th_f_out = 0.8;
R_th_s_out = 0.3;

R_th_c = 2.0;
R_th_s = 0.3;

%% --- Print header ---
fprintf('========================================================\n');
fprintf(' MISO RIS-ISAC-NOMA v3-EASY  (M=%d, N=%d, samples=%d)\n', M, N, num_samples);
fprintf('========================================================\n');
fprintf('Path loss (dB):\n');
fprintf('  BS->RIS=%.2f  RIS->Dn=%.2f  RIS->Df=%.2f\n', ...
        10*log10(PL_BR), 10*log10(PL_RDn), 10*log10(PL_RDf));
fprintf('  RIS->T =%.2f  T->RIS =%.2f\n', ...
        10*log10(PL_RT), 10*log10(PL_TR));
fprintf('Power:  P_tot=%d dBm  sigma^2=%.2f dBm  SNR budget=%.2f dB\n', ...
        P_tot_dBm, 10*log10(sigma2)+30, 10*log10(P_tot/sigma2));
fprintf('NOMA:   a_n=%.2f  a_f=%.2f  a_T=%.2f  (perfect SIC)\n\n', a_n, a_f, a_T);

%% --- Storage ---
H_BR_all   = zeros(N, M, num_samples);
h_RDn_all  = zeros(N, 1, num_samples);
h_RDf_all  = zeros(N, 1, num_samples);
h_RT_all   = zeros(N, 1, num_samples);
h_TR_all   = zeros(N, 1, num_samples);
h_BDn_all  = zeros(M, 1, num_samples);
h_BDf_all  = zeros(M, 1, num_samples);
Theta_all  = zeros(N, N, num_samples);

w_c_all = zeros(M, 1, num_samples);
w_T_all = zeros(M, 1, num_samples);
u_T_all = zeros(M, 1, num_samples);

SINR_f_all = zeros(1, num_samples);
SINR_n_all = zeros(1, num_samples);
SINR_s_all = zeros(1, num_samples);
R_f_all    = zeros(1, num_samples);
R_n_all    = zeros(1, num_samples);
R_s_all    = zeros(1, num_samples);

out_f_sic = 0; out_n_sic = 0; out_s_sic = 0;
out_f_rate = 0; out_n_rate = 0; out_s_rate = 0; out_any_rate = 0;

qos_viol_c_all = false(1, num_samples);
qos_viol_s_all = false(1, num_samples);
qos_viol_any   = false(1, num_samples);

%% ===================== MAIN MONTE CARLO LOOP =====================
for s = 1:num_samples

    % 1. Generate channels
    H_BR  = nakagami_channel(N, M, m_BR,  PL_BR);
    h_RDn = nakagami_channel(N, 1, m_RDn, PL_RDn);
    h_RDf = nakagami_channel(N, 1, m_RDf, PL_RDf);
    h_RT  = nakagami_channel(N, 1, m_RT,  PL_RT);
    h_TR  = nakagami_channel(N, 1, m_TR,  PL_TR);

    if include_direct
        h_BDn = nakagami_channel(M, 1, m_BDn, PL_BDn);
        h_BDf = nakagami_channel(M, 1, m_BDf, PL_BDf);
    else
        h_BDn = zeros(M, 1);
        h_BDf = zeros(M, 1);
    end

    % 2. Random continuous RIS phases
    theta_raw = 2*pi * rand(N, 1);
    Theta     = eta_ris * diag(exp(1j * theta_raw));

    % 3. Cascaded effective MISO channels
    h_n = (h_RDn.' * Theta * H_BR).' + h_BDn;
    h_f = (h_RDf.' * Theta * H_BR).' + h_BDf;

    % 4. Monostatic sensing channel G_T (M x M)
    G_T = sqrt(beta_T) * (H_BR' * Theta * h_TR) * (h_RT.' * Theta * H_BR);

    % 5. NOMA beamforming: cluster MRT on far user, null-space sensing BF
    w_c = h_f / max(norm(h_f), 1e-15);   % shared for both comm users

    H_u    = [h_n, h_f];
    P_null = eye(M) - H_u / (H_u' * H_u) * H_u';
    A_proj = P_null * (G_T' * G_T) * P_null;
    A_proj = (A_proj + A_proj') / 2;
    [V_A, D_A] = eig(A_proj);
    [~, idx_max] = max(real(diag(D_A)));
    w_T = V_A(:, idx_max);
    w_T = w_T / max(norm(w_T), 1e-15);

    g_T_eff = G_T * w_T;
    if norm(g_T_eff) > 1e-15
        u_T = g_T_eff / norm(g_T_eff);
    else
        u_T = zeros(M, 1); u_T(1) = 1;
    end

    % 6. Effective gains (w_n = w_f = w_c so gff = gfn)
    gff = abs(h_f' * w_c)^2;
    gfn = abs(h_f' * w_c)^2;   % = gff (same beamformer)
    gfT = abs(h_f' * w_T)^2;

    gnn = abs(h_n' * w_c)^2;
    gnT = abs(h_n' * w_T)^2;

    gsT = abs(u_T' * G_T * w_T)^2;
    gsf = abs(u_T' * G_T * w_c)^2;
    gsn = abs(u_T' * G_T * w_c)^2;   % = gsf (same beamformer)

    % 7. SINRs (perfect SIC: delta = 0)
    snr_f = (gff * a_f * P_tot) / ...
            (gfn * a_n * P_tot + gfT * a_T * P_tot + sigma2);

    snr_n = (gnn * a_n * P_tot) / ...
            (delta_f * abs(h_n' * w_c)^2 * a_f * P_tot + gnT * a_T * P_tot + sigma2);

    snr_s = (gsT * a_T * P_tot) / ...
            (delta_f * gsf * a_f * P_tot + delta_n * gsn * a_n * P_tot + sigma2);

    snr_n_to_f = (abs(h_n' * w_c)^2 * a_f * P_tot) / ...
                 (gnn * a_n * P_tot + gnT * a_T * P_tot + sigma2);

    % 8. Outage checks
    if snr_f < gamma_c_th
        out_f_sic = out_f_sic + 1;
    end
    if ~(snr_n_to_f > gamma_SIC_th && snr_n > gamma_c_th)
        out_n_sic = out_n_sic + 1;
    end
    if snr_s < gamma_s_th
        out_s_sic = out_s_sic + 1;
    end

    % 9. Store
    H_BR_all (:,:,s) = H_BR;
    h_RDn_all(:,:,s) = h_RDn;
    h_RDf_all(:,:,s) = h_RDf;
    h_RT_all (:,:,s) = h_RT;
    h_TR_all (:,:,s) = h_TR;
    h_BDn_all(:,:,s) = h_BDn;
    h_BDf_all(:,:,s) = h_BDf;
    Theta_all(:,:,s) = Theta;

    w_c_all(:,:,s) = w_c;
    w_T_all(:,:,s) = w_T;
    u_T_all(:,:,s) = u_T;

    SINR_f_all(s) = snr_f;
    SINR_n_all(s) = snr_n;
    SINR_s_all(s) = snr_s;
    R_f_all(s)    = log2(1 + snr_f);
    R_n_all(s)    = log2(1 + snr_n);
    R_s_all(s)    = log2(1 + snr_s);

    fail_n = (R_n_all(s) < R_th_n_out);
    fail_f = (R_f_all(s) < R_th_f_out);
    fail_s = (R_s_all(s) < R_th_s_out);
    if fail_n, out_n_rate = out_n_rate + 1; end
    if fail_f, out_f_rate = out_f_rate + 1; end
    if fail_s, out_s_rate = out_s_rate + 1; end
    if fail_n || fail_f || fail_s, out_any_rate = out_any_rate + 1; end

    R_comm_sum      = R_n_all(s) + R_f_all(s);
    qos_viol_c_all(s) = (R_comm_sum  < R_th_c);
    qos_viol_s_all(s) = (R_s_all(s)  < R_th_s);
    qos_viol_any(s)   = qos_viol_c_all(s) || qos_viol_s_all(s);
end

%% ===================== AGGREGATE RESULTS =====================
w_c_obj = 0.7;  w_s_obj = 0.3;
R_DL_sum_avg = mean(w_c_obj * (R_n_all + R_f_all) + w_s_obj * R_s_all);

fprintf('=== Mean SINR (dB) ===\n');
fprintf('  Near User : %.2f\n', 10*log10(mean(SINR_n_all)+1e-12));
fprintf('  Far  User : %.2f\n', 10*log10(mean(SINR_f_all)+1e-12));
fprintf('  Sensing   : %.2f\n', 10*log10(mean(SINR_s_all)+1e-12));

fprintf('\n=== Average rates (%d samples) ===\n', num_samples);
fprintf('  Near User R_n : %.4f bits/s/Hz\n', mean(R_n_all));
fprintf('  Far  User R_f : %.4f bits/s/Hz\n', mean(R_f_all));
fprintf('  Sensing   R_s : %.4f bits/s/Hz\n', mean(R_s_all));
fprintf('  R_DL_sum (w_c=%.2f, w_s=%.2f) : %.4f bits/s/Hz\n\n', ...
        w_c_obj, w_s_obj, R_DL_sum_avg);

fprintf('=== Rate-outage probabilities ===\n');
fprintf('  Near : %.4f  Far : %.4f  Sensing : %.4f  Any : %.4f\n\n', ...
    out_n_rate/num_samples, out_f_rate/num_samples, ...
    out_s_rate/num_samples, out_any_rate/num_samples);

fprintf('=== Decode-chain outage (SIC+SINR thresholds) ===\n');
fprintf('  Far : %.4f  Near : %.4f  Sensing : %.4f\n\n', ...
    out_f_sic/num_samples, out_n_sic/num_samples, out_s_sic/num_samples);

P_qos_c   = mean(qos_viol_c_all);
P_qos_s   = mean(qos_viol_s_all);
P_qos_any = mean(qos_viol_any);
fprintf('=== QoS violation (R_th_c=%.2f, R_th_s=%.2f) ===\n', R_th_c, R_th_s);
fprintf('  Comm : %.4f  Sensing : %.4f  Any : %.4f\n\n', P_qos_c, P_qos_s, P_qos_any);
fprintf('  Baseline R_DL_sum = %.4f bits/s/Hz  QoS viol = %.1f%%\n\n', ...
        R_DL_sum_avg, 100*P_qos_any);

%% ===================== VISUALISATION =====================
SINR_f_dB = 10*log10(SINR_f_all + 1e-12);
SINR_n_dB = 10*log10(SINR_n_all + 1e-12);
SINR_s_dB = 10*log10(SINR_s_all + 1e-12);

figure('Name','MISO RIS-ISAC-NOMA v3-EASY','NumberTitle','off', ...
       'Position',[60 60 1200 800]);

subplot(2,3,1);
histogram(SINR_n_dB, 40, 'FaceColor','#D95319','EdgeColor','none');
title('SINR Near User'); xlabel('SINR (dB)'); ylabel('Count'); grid on;

subplot(2,3,2);
histogram(SINR_f_dB, 40, 'FaceColor','#EDB120','EdgeColor','none');
title('SINR Far User'); xlabel('SINR (dB)'); ylabel('Count'); grid on;

subplot(2,3,3);
histogram(SINR_s_dB, 40, 'FaceColor','#77AC30','EdgeColor','none');
title('SINR Sensing'); xlabel('SINR (dB)'); ylabel('Count'); grid on;

subplot(2,3,4);
histogram(R_n_all, 40, 'FaceColor','#D95319','EdgeColor','none');
title('Rate Near User'); xlabel('bits/s/Hz'); ylabel('Count'); grid on;

subplot(2,3,5);
histogram(R_f_all, 40, 'FaceColor','#EDB120','EdgeColor','none');
title('Rate Far User'); xlabel('bits/s/Hz'); ylabel('Count'); grid on;

subplot(2,3,6);
histogram(R_s_all, 40, 'FaceColor','#77AC30','EdgeColor','none');
title('Rate Sensing'); xlabel('bits/s/Hz'); ylabel('Count'); grid on;

sgtitle('MISO RIS-ISAC-NOMA v3-EASY (OMA\_new folder, MRT baseline)','FontSize',13);

plot_file = fullfile(pwd, 'MISO_ISAC_NOMA_v3_easy_Results.png');
try
    exportgraphics(gcf, plot_file, 'Resolution', 300);
catch
    saveas(gcf, plot_file);
end
fprintf('Plot saved -> %s\n', plot_file);

%% ===================== SAVE DATASET =====================
output_mat = fullfile(pwd, 'ISAC_RIS_NOMA_channels_v3_easy.mat');
save(output_mat, ...
    'H_BR_all','h_RDn_all','h_RDf_all','h_RT_all','h_TR_all', ...
    'h_BDn_all','h_BDf_all','Theta_all', ...
    'w_c_all','w_T_all','u_T_all', ...
    'SINR_f_all','SINR_n_all','SINR_s_all', ...
    'R_f_all','R_n_all','R_s_all', ...
    'qos_viol_c_all','qos_viol_s_all','qos_viol_any', ...
    'R_th_c','R_th_s','R_th_n_out','R_th_f_out','R_th_s_out', ...
    'gamma_c_th','gamma_s_th','gamma_SIC_th', ...
    'N','M','num_samples', ...
    'a_n','a_f','a_T','P_tot','sigma2','delta_f','delta_n', ...
    'w_c_obj','w_s_obj','PL_BR','PL_RDn','PL_RDf','PL_RT','PL_TR', ...
    'PL_BDn','PL_BDf','beta_T','eta_ris','NF_dB','include_direct','-v7.3');

fprintf('Dataset saved -> %s\n', output_mat);

%% ===================== LOCAL FUNCTIONS =====================
function H = nakagami_channel(rows, cols, m_val, path_loss)
    m_int = round(m_val);
    assert(abs(m_val - m_int) < 1e-12 && m_int >= 1, 'm must be positive integer.');
    expo   = -log(rand(rows, cols, m_int));
    amp_sq = sum(expo, 3) / m_val;
    amp    = sqrt(amp_sq);
    phase  = 2*pi * rand(rows, cols);
    H      = sqrt(path_loss) * amp .* exp(1j*phase);
end

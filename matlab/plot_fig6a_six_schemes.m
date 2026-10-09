%% =========================================================
%  plot_fig6a_six_schemes.m
%  ---------------------------------------------------------
%  Renders the weighted sum-rate CDF across six NOMA schemes in
%  MATLAB, reading the per-channel data produced by
%  fig6_six_scheme_compare.py (results/csv/fig6_six_scheme_percurve.csv).
%  Use this if you prefer a MATLAB-native figure instead of / in
%  addition to the Python-rendered PNG/PDF.
% =========================================================

clc; clear; close all;

here = fileparts(mfilename('fullpath'));
repo_root = fileparts(here);
csv_path = fullfile(repo_root, 'results', 'csv', 'fig6_six_scheme_percurve.csv');
out_dir  = fullfile(repo_root, 'results', 'figures');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

T = readtable(csv_path);
names = T.Properties.VariableNames;

% Column order written by the Python script:
legend_labels = { ...
    'Proposed (RIS+ML optimization)', ...
    'Equal power + random RIS phases', ...
    'Random power + random RIS phases', ...
    'Equal power + fixed RIS phases', ...
    'Random power + fixed RIS phases', ...
    'NOMA + No-RIS'};

colors = { [0.84 0.15 0.16], [0.12 0.47 0.71], [0.58 0.40 0.74], ...
           [0.17 0.63 0.17], [0.55 0.34 0.29], [0.33 0.33 0.33] };
styles = {'-','-','-','--','--',':'};
widths = [2.6, 1.8, 1.8, 1.8, 1.8, 2.2];

figure('Position',[100 100 760 540]); hold on; grid on; box on;
for i = 1:numel(names)
    col = T.(names{i});
    col = col(~isnan(col));
    xs = sort(col);
    ys = linspace(0,1,numel(xs));
    plot(xs, ys, 'Color', colors{i}, 'LineStyle', styles{i}, ...
         'LineWidth', widths(i), 'DisplayName', legend_labels{i});
end
xlabel('Weighted sum rate R_{DL} (bits/s/Hz)', 'FontSize', 12);
ylabel('CDF', 'FontSize', 12);
title('Weighted sum-rate CDF across NOMA schemes', 'FontSize', 12);
legend('Location','southeast','FontSize',8);

out_file = fullfile(out_dir, 'fig6a_six_schemes_matlab.png');
exportgraphics(gcf, out_file, 'Resolution', 300);
fprintf('Saved -> %s\n', out_file);

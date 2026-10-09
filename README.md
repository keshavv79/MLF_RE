# Meta-Learning-Driven Joint Power Allocation and Phase-Shift Design for RIS-Assisted NOMA-ISAC Systems

Code, trained models, and results for a MISO RIS-assisted NOMA-ISAC downlink in
which a base station jointly serves a near and a far user via NOMA while
illuminating a sensing target. A MAML-based deep-learning solver learns the
NOMA power split while refining the RIS phase shifts per channel realization
with an unrolled inner loop.

## Repository layout

```
src/                 Python: trainers, evaluators, plotting utilities
matlab/              MATLAB channel-dataset generators (.m)
notebooks/           Kaggle notebooks for training on GPU
models/              Trained checkpoints (.pt)
results/             Final figures (.jpg/.png) and summary CSVs
data/                Datasets are large and not tracked -- see data/README.md
```

## Setup

```bash
pip install torch h5py numpy matplotlib --break-system-packages
```

Datasets (`.mat`) are large (~1.8 GB total) and are not stored in the repo.
Regenerate them with the MATLAB scripts in `matlab/`, or place downloaded
copies in `data/`. See `data/README.md`.

## Method at a glance

The base station transmits a superposed signal with a three-way power split
`a = (a_n, a_f, a_T)` (near / far / sensing) and a shared cluster beamformer,
through an N-element RIS with phase matrix `Phi`. The solver:

- **Power split `a`** -- predicted by a small MLP with a softmax head (stays
  on the simplex by construction).
- **RIS phase `phi`** -- refined per channel by K steps of projected gradient
  ascent (inner loop).
- **Meta-training** -- the network weights are updated across a batch of
  channels with a differentiable Lagrangian that penalizes QoS violations.

Fixed closed-form beamformers are used for the cluster (`w_c`) and sensing
(`w_T`) directions; the learned/optimized quantities are the power split and
the RIS phase.

## System and training configuration

- BS antennas M=4, RIS elements N=8 (swept to 16/32/64 for the rate-vs-N
  figure).
- Nominal transmit power: 15 dBm (swept 5/10/15/20/25 dBm for the
  rate-vs-power figure).
- Target path-gain `beta_T`: 65 dB.
- NOMA-RIS QoS floors: `R_th,c = R_th,s = 3.1` bit/s/Hz.
- OMA-RIS QoS floors: `R_th,c = 3.56`, `R_th,s = 1.79` bit/s/Hz.
- No-RIS QoS floors (both NOMA and OMA): `R_th,c = 2.0`, `R_th,s = 0.3`
  bit/s/Hz.

## Code (`src/`)

**Trainers** (each writes a `.pt` checkpoint):

| Script | Trains | Dataset | Output |
|---|---|---|---|
| `train_noma_fair_maml.py` | NOMA + RIS (main model) | `ISAC_RIS_NOMA_channels_v3_fair.mat` | `policy_noma_fair_best.pt` |
| `train_noma_fair_beam_maml.py` | NOMA + RIS, learns cluster-beam mix | same | `policy_noma_fair_beam_best.pt` |
| `train_noma_fair_noris.py` | NOMA, no RIS | `ISAC_NOMA_channels_fair_noris.mat` | `policy_noris_best.pt` |
| `train_dl_oma_maml.py` | OMA + RIS | `ISAC_RIS_OMA_channels_v3_fair.mat` | `policy_oma_fair_best.pt` |
| `train_noris_oma.py` | OMA, no RIS | `ISAC_OMA_channels_fair_noris.mat` | `policy_noris_oma_best.pt` |

`train_dl_v3_easy.py` is a shared library (dataset loader + base network)
imported by the others -- not run directly.

**Evaluators / plotting:**

- `compare_noma_fair.py` -- runs the NOMA ablation, produces the sum-rate CDF,
  the QoS-violation breakdown, and summary CSVs.
- `compare_oma_v3_easy.py` -- OMA counterpart.
- `plot_fig5_4curve_FIXED.py` -- builds the 4-curve rate-vs-N figure (NOMA-RIS,
  NOMA no-RIS, OMA-RIS, OMA no-RIS) by reading checkpoints directly from
  `models/noma-fair/`, `models/oma_fair/`, and `models/no-ris/`. Writes
  `fig5_4curve_FIXED.png`.
- `plot_fig8_4curve.py`, `make_combined_figs.py` -- build the rate-vs-power
  figure.
- `make_loss_curves.py`, `render_fair_plots.py`, `make_fair_plots.py` -- render
  training-loss and summary figures.
- `rebuild_fig8_csv.py`, `sanity_check.py` -- helpers.
- `gen_fig6_channels.py`, `fig6_six_scheme_compare.py`,
  `plot_fig6a_six_schemes.py` -- the six-scheme NOMA comparison pipeline,
  described below.

## Trained models (`models/`)

```
models/
  noma-fair/
    policy_noma_fair_best.pt          main proposed model (NOMA + RIS)
    policy_noma_fair_beam_best.pt     beam-learning variant (secondary)
    fig-5/  N16,N32,N64 checkpoints    rate-vs-N sweep
    fig-8/  ris/ , no_ris/             rate-vs-power sweep
  oma_fair/
    policy_oma_fair_best.pt + fig5_oma/ + fig8_oma/
  no-ris/
    policy_noris_best.pt              NOMA, no RIS
    policy_noris_oma_best.pt          OMA, no RIS
```

`policy_noris_best.pt` and `policy_noris_oma_best.pt` are trained on the M=4
"fair" no-RIS datasets (`ISAC_NOMA_channels_fair_noris.mat`,
`ISAC_OMA_channels_fair_noris.mat`), matching the scenario used by every other
checkpoint in this repo:

```bash
python train_noma_fair_noris.py --mat ISAC_NOMA_channels_fair_noris.mat \
  --P_tot_dBm 15 --epochs 60 --batch 128 --out policy_noris_best.pt

python train_noris_oma.py --mat ISAC_OMA_channels_fair_noris.mat \
  --P_tot_dBm 15 --epochs 60 --batch 128 --out policy_noris_oma_best.pt
```

## Six-scheme NOMA comparison

Compares six NOMA configurations on the same held-out test channels so the
comparison is apples-to-apples:

1. **Proposed (RIS + ML optimization)** -- the main trained policy
   (`policy_noma_fair_best.pt`), run through its K-step inner-loop RIS-phase
   adaptation.
2. **Equal power + random RIS phases** -- `a=(1/3,1/3,1/3)`, fresh i.i.d.
   `Uniform(0,2*pi)` RIS phase per channel.
3. **Random power + random RIS phases** -- `softmax(N(0,1))` power draw per
   channel, fresh random RIS phase per channel.
4. **Equal power + fixed RIS phases** -- `a=(1/3,1/3,1/3)`, a single
   non-reconfigured RIS profile (`Theta = I`) applied to every channel.
5. **Random power + fixed RIS phases** -- `softmax(N(0,1))` power draw, same
   fixed `Theta = I` profile.
6. **NOMA + No-RIS** -- the trained no-RIS policy (`policy_noris_best.pt`) on
   direct BS-to-user/target links (no RIS in the scenario at all).

All six schemes share the same closed-form beamformers (fair-weighted common
beam `w_c = 0.7*h_f + 0.3*h_n`, soft-null sensing beam `w_T`) and the paper's
weighted objective `R_DL = 0.7*(R_n+R_f) + 0.3*R_s`.

**Files:**

- `src/gen_fig6_channels.py` (+ `matlab/gen_fig6_ris_test_channels.m` and
  `matlab/gen_fig6_noris_test_channels.m`) -- regenerates the two held-out
  test-channel sets used for this comparison. The RIS-branch channels mirror
  `matlab/miso_isac_noma_v3_fair_chatpgt.m`; the no-RIS-branch channels mirror
  the M=4 "fair" no-RIS scenario (see the script's docstring for the exact
  path-loss exponent and blockage-factor values -- note the NOMA and OMA
  no-RIS "fair" scenarios use different blockage factors from each other,
  29 dB vs. 42 dB, so they are not interchangeable). The Python script writes
  `.npz` files alongside itself in `src/`; the MATLAB scripts write `.mat`
  files to the current working directory.
- `src/fig6_six_scheme_compare.py` -- the main evaluator. Loads the trained
  checkpoints and physics functions directly from `src/` and `models/` (no
  reimplementation), evaluates all six schemes, and writes
  `results/csv/fig6_six_scheme_summary.csv` (per-scheme mean rates and
  QoS-violation rate) and `results/csv/fig6_six_scheme_percurve.csv`
  (per-channel data for the CDF).
- `src/plot_fig6a_six_schemes.py` / `matlab/plot_fig6a_six_schemes.m` --
  renders the CDF from the per-curve CSV, producing
  `results/figures/fig6a_six_schemes.png` / `.pdf`.

**Results:**

| Scheme | R_DL_sum (bits/s/Hz) | QoS-violation rate |
|---|---|---|
| Proposed (RIS+ML optimization) | **11.02** | **1.2%** |
| Equal power + random RIS phases | 10.79 | 7.4% |
| Equal power + fixed RIS phases | 10.79 | 7.3% |
| Random power + random RIS phases | 10.49 | 18.4% |
| Random power + fixed RIS phases | 10.49 | 18.4% |
| NOMA + No-RIS | 6.69 | 5.3% |

The proposed scheme gives the highest weighted rate and the lowest
QoS-violation rate; random power allocation is clearly worse than equal power
(it is not tuned to the NOMA/sensing constraints); RIS phase randomness vs. a
fixed phase barely matters once the power split is held fixed; and NOMA
without RIS is clearly worst on weighted rate, confirming RIS assistance is
necessary at this operating point, not just power/phase optimization alone.

To re-run:

```bash
cd src
python gen_fig6_channels.py          # regenerate held-out test channels (optional, already included)
python fig6_six_scheme_compare.py    # writes the two CSVs to results/csv/
python plot_fig6a_six_schemes.py     # writes fig6a_six_schemes.png/.pdf to results/figures/
```

Note: `.npz` channel files produced by `gen_fig6_channels.py` are not tracked
in this repo (same reasoning as `data/` -- they're deterministic given the
fixed RNG seeds in the script, so anyone can regenerate them in seconds).

## Typical workflow

1. Generate datasets: run the scripts in `matlab/` (or download into `data/`).
2. Train: `python src/train_noma_fair_maml.py` (or use the Kaggle notebook).
3. Evaluate: `python src/compare_noma_fair.py --ckpt_maml models/noma-fair/policy_noma_fair_best.pt`
4. Plot sweeps: `python src/plot_fig5_4curve_FIXED.py`, `python src/plot_fig8_4curve.py`
5. Six-scheme comparison: see "Six-scheme NOMA comparison" above.

## Results (`results/`)

`results/figures/` holds the rendered figures; `results/csv/` holds the
numeric summaries (ablation table, rate-vs-N, rate-vs-power).

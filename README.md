# Meta-Learning-Driven Joint Power Allocation and Phase-Shift Design for RIS-Assisted NOMA-ISAC Systems

Code, trained models, and results for a MISO RIS-assisted NOMA-ISAC downlink in
which a base station jointly serves a near and a far user via NOMA while
illuminating a sensing target. A MAML-based deep-learning solver learns the NOMA
power split while refining the RIS phase shifts per channel realization with an
unrolled inner loop.

## Repository layout

```
src/         Python: trainers, evaluators, plotting utilities
matlab/      MATLAB channel-dataset generators (.m)
notebooks/   Kaggle notebooks for training on GPU
models/      Trained checkpoints (.pt)
results/     Final figures (.jpg) and summary CSVs
data/        Datasets are large and not tracked — see data/README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Datasets (`.mat`) are large (~1.8 GB total) and are not stored in the repo.
Regenerate them with the MATLAB scripts in `matlab/`, or place downloaded copies
in `data/`. See `data/README.md`.

## Method at a glance

The base station transmits a superposed signal with a three-way power split
`a = (a_n, a_f, a_T)` (near / far / sensing) and a shared cluster beamformer,
through an N-element RIS with phase matrix `Phi`. The solver:

- **Power split `a`** — predicted by a small MLP with a softmax head (stays on the
  simplex by construction).
- **RIS phase `phi`** — refined per channel by K steps of projected gradient
  ascent (inner loop).
- **Meta-training** — the network weights are updated across a batch of channels
  with a differentiable Lagrangian that penalizes QoS violations.

Fixed closed-form beamformers are used for the cluster (`w_c`) and sensing (`w_T`)
directions; the learned/optimized quantities are the power split and the RIS
phase.

## Code (`src/`)

**Trainers** (each writes a `.pt` checkpoint):

| Script | Trains | Dataset | Output |
|---|---|---|---|
| `train_noma_fair_maml.py` | NOMA + RIS (main model) | `ISAC_RIS_NOMA_channels_v3_fair.mat` | `policy_noma_fair_best.pt` |
| `train_noma_fair_beam_maml.py` | NOMA + RIS, learns cluster-beam mix | same | `policy_noma_fair_beam_best.pt` |
| `train_noma_fair_noris.py` | NOMA, no RIS | `ISAC_NOMA_channels_fair_noris.mat` | `policy_noma_fair_noris_best.pt` |
| `train_dl_oma_maml.py` | OMA + RIS | `ISAC_RIS_OMA_channels_v3_fair.mat` | `policy_oma_fair_best.pt` |
| `train_noris_oma.py` | OMA, no RIS | `ISAC_OMA_channels_noris.mat` | `policy_noris_oma_best.pt` |

`train_dl_v3_easy.py` is a shared library (dataset loader + base network) imported
by the others — not run directly.

**Evaluators / plotting:**

- `compare_noma_fair.py` — runs the NOMA ablation, produces the sum-rate CDF, the
  QoS-violation breakdown, and summary CSVs.
- `compare_oma_v3_easy.py` — OMA counterpart.
- `plot_fig5_4curve.py`, `plot_fig8_4curve.py`, `make_combined_figs.py` — build the
  rate-vs-N and rate-vs-power figures.
- `make_loss_curves.py`, `render_fair_plots.py`, `make_fair_plots.py` — render
  training-loss and summary figures.
- `rebuild_fig8_csv.py`, `sanity_check.py` — helpers.

## Trained models (`models/`)

```
models/
  noma_fair/
    policy_noma_fair_best.pt          main proposed model
    policy_noma_fair_beam_best.pt     beam-learning variant
    fig5/  N16,N32,N64 checkpoints     rate-vs-N sweep
    fig8/  ris/ , no_ris/              rate-vs-power sweep
  oma_fair/
    policy_oma_fair_best.pt + fig5_oma/ + fig8_oma/
  no_ris/
    policy_noris_best.pt              NOMA no-RIS
    policy_noris_oma_best.pt          OMA no-RIS
```

## Typical workflow

1. Generate datasets: run the scripts in `matlab/` (or download into `data/`).
2. Train: `python src/train_noma_fair_maml.py` (or use the Kaggle notebook).
3. Evaluate: `python src/compare_noma_fair.py --ckpt_maml models/noma_fair/policy_noma_fair_best.pt`
4. Plot sweeps: `python src/plot_fig5_4curve.py`, `python src/plot_fig8_4curve.py`

## Results (`results/`)

`results/figures/` holds the rendered figures; `results/csv/` holds the numeric
summaries (ablation table, rate-vs-N, rate-vs-power).

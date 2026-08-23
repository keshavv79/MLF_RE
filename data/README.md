# Datasets

The channel datasets (`.mat`) are large (~1.8 GB total) and are **not** stored in
this repository. Two ways to obtain them:

## Option 1 — Regenerate (MATLAB)

Run the generators in `../matlab/`:

- `miso_isac_noma_v3_fair_chatpgt.m` — main NOMA + RIS channels
  (`ISAC_RIS_NOMA_channels_v3_fair.mat`)
- `miso_isac_oma_v3_fair_chatpgt.m` — OMA + RIS channels
- `gen_oma_all_N.m` — the N-sweep datasets (N = 16, 32, 64)
- `gen_test_set_oma.m` — held-out test sets
- `gen_noris_oma_dataset.m` — no-RIS channels

Place the generated `.mat` files in this `data/` directory.

## Option 2 — Download

Place pre-generated `.mat` files here. Expected files:

```
ISAC_RIS_NOMA_channels_v3_fair.mat          main NOMA training set
ISAC_RIS_NOMA_channels_v3_fair_TEST.mat     NOMA test set
ISAC_RIS_NOMA_channels_v3_fair_N16/N32/N64.mat
ISAC_NOMA_channels_fair_noris.mat           NOMA no-RIS
ISAC_RIS_OMA_channels_v3_fair*.mat          OMA counterparts
ISAC_OMA_channels_fair_noris.mat            OMA no-RIS
```

The training and evaluation scripts read from paths passed via `--mat`; point them
at the files in this directory.

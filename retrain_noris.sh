#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
NM=m4_datasets/ISAC_NOMA_channels_fair_noris.mat
OM=m4_datasets/ISAC_OMA_channels_fair_noris.mat
for P in 35 37.5 40 42.5 45; do
  echo "===== NOMA no-RIS  P=$P ====="
  python train_noma_fair_noris.py --mat "$NM" --P_tot_dBm $P \
    --out "_fair_noma/fig8/no_ris/noris_P${P}.pt" --epochs 60 2>&1 | tail -2
  echo "===== OMA no-RIS   P=$P ====="
  python train_noris_oma.py --mat "$OM" --P_tot_dBm $P --lam_s 150 \
    --out "_fair_oma/fig8_oma/no_ris/noris_oma_P${P}.pt" --epochs 60 2>&1 | tail -2
done
echo "ALL_NORIS_DONE"

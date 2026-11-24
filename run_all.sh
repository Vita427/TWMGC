#!/usr/bin/env bash
# Runs every dataset of the paper with ten seeds, as reported in Table 2.
set -e

DATASETS="acm amap amac cocs roman_empire amazon_ratings"

for ds in ${DATASETS}; do
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== ${ds}, seed ${seed}"
    python main.py --name "${ds}" --seed "${seed}"
  done
done

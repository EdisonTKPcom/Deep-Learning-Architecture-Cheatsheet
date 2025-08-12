#!/usr/bin/env bash
set -euo pipefail

python -m dlx.train \
  --dataset cifar10 \
  --model cnn_small \
  --epochs 10 \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --device cpu \
  --num-workers 0
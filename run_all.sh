#!/bin/bash
set -euo pipefail # if crash close programmmm

mkdir -p results checkpoints

# 1. Teacher
python train_vit_l.py        2>&1 | tee results/log_vit_l.txt
# 2. Student baseline
python train_vit_t.py        2>&1 | tee results/log_vit_t.txt
# 3. Classical KD
python train_kd_classical.py 2>&1 | tee results/log_kd.txt
# 4. DeiT-style distillation
python train_deit_distill.py 2>&1 | tee results/log_deit.txt
# 5. Evaluate + compare
python evaluate_all.py       2>&1 | tee results/log_eval.txt

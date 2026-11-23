#!/bin/bash
#SBATCH -A info7375.202710 -p gpu,gpu-short --gres=gpu:h200:1 -t 01:00:00 -c 16 --mem=64G
#SBATCH -o /scratch/%u/info7375-w9/logs/%x-%j.out
# sbatch -J eval cluster/eval.sbatch --mode agentflow --out results/agentflow.json [--adapter ...]
# (fits on an 80 GB GPU too: add --gres=gpu:h100:1 before the script name)
set -u
source /scratch/$USER/info7375-w9/code/cluster/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python evaluate.py "$@"

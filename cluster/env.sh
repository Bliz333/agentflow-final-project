# Sourced by the Slurm scripts. Paths assume the layout in ../README.md: code in $W/code, data in $W/data.
W=/scratch/$USER/info7375-w9
export HF_HOME=/scratch/$USER/hf-home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/scratch/$USER/envs/agentflow/bin:$PATH
set -a; for f in ~/.config/info7375/*.env; do . $f; done; set +a  # judge.env: OPENCODE_GO_API_KEY, deepseek.env: DEEPSEEK_API_KEY
cd $W/code

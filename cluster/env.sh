# Sourced by the Slurm scripts. Paths assume the layout in ../README.md: code in $W/code, data in $W/data.
W=/scratch/$USER/info7375-w9
export HF_HOME=/scratch/$USER/hf-home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/scratch/$USER/envs/w8/bin:$PATH
set -a; . ~/.config/info7375/judge.env; set +a  # OPENCODE_GO_API_KEY for the judge
cd $W/code

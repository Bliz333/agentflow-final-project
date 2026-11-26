"""Run evaluate.py and train.py on Modal: one GPU, the model weights, Search-R1's retrieval data and the
Flow-GRPO runs in Modal volumes, the evaluation's result file returned.

    modal run modal_app.py::check                     # CPU only: build the image, run the tests
    modal run modal_app.py::prep                      # CPU only, once: retrieval data and training questions
    modal run modal_app.py --args "--mode agentflow --search wikipedia --limit 5 --trials 1" --out results/smoke.json
    modal run modal_app.py::train --args "--out /runs/flow_grpo_google --resume --steps 30 --google-search"

Evaluations need the judge key (OPENCODE_GO_API_KEY) in the local environment, and --google-search also the
DeepSeek key (DEEPSEEK_API_KEY): the keys set locally are passed to the run and not stored in the Modal workspace.
"""
import os
import shlex
import subprocess
from pathlib import Path

import modal

CODE = "/root/code"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0+cu129", "torchvision==0.28.0+cu129", "torchaudio==2.11.0+cu129",
                    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/"
                    "vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
                    "transformers==5.19.0", "peft==0.21.2", "openai", "pandas", "pyarrow", "pytest",
                    extra_index_url="https://download.pytorch.org/whl/cu129",
                    extra_options="--index-strategy unsafe-best-match")
    .run_commands("python -m pip uninstall -y torchcodec")  # vLLM's video decoder: unused, and its libraries fail to load
    .env({"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(Path(__file__).parent, CODE, ignore=["data", "runs", "results", ".git", "**/__pycache__",
                                                       ".pytest_cache"])
)
weights = modal.Volume.from_name("agentflow-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("agentflow-data", create_if_missing=True)  # /data: retrieval/, nq_hotpotqa_train/
runs = modal.Volume.from_name("agentflow-runs", create_if_missing=True)  # /runs: Flow-GRPO checkpoints, adapters
VOLUMES = {"/root/.cache/huggingface": weights, "/data": data, "/runs": runs}
KEYS = [modal.Secret.from_local_environ([k]) for k in ["OPENCODE_GO_API_KEY", "DEEPSEEK_API_KEY"] if os.environ.get(k)]
app = modal.App("agentflow-final-project", image=image)


@app.function(timeout=900)
def check():
    subprocess.run(["python", "-m", "pytest", "-q", "test_agentflow.py"], cwd=CODE, check=True)


@app.function(cpu=8, memory=16384, ephemeral_disk=512 * 1024, timeout=3 * 3600, volumes={"/data": data})
def prep():
    """Search-R1's files from Hugging Face, converted by prep_index.py into /data/retrieval (45 GB)."""
    from huggingface_hub import snapshot_download
    for repo, local in [("PeterJinGo/wiki-18-e5-index", "/tmp/index"), ("PeterJinGo/wiki-18-corpus", "/tmp/corpus"),
                        ("PeterJinGo/nq_hotpotqa_train", "/data/nq_hotpotqa_train")]:
        snapshot_download(repo, repo_type="dataset", local_dir=local)
    subprocess.run(["python", "prep_index.py", "--index-dir", "/tmp/index", "--corpus", "/tmp/corpus/wiki-18.jsonl.gz",
                    "--out", "/data/retrieval"], cwd=CODE, check=True)
    data.commit()


@app.function(gpu="H200", timeout=4 * 3600, volumes=VOLUMES, secrets=KEYS)
def train(args: str):
    try:
        subprocess.run(["python", "train.py", "--retrieval", "/data/retrieval", "--data",
                        "/data/nq_hotpotqa_train/train.parquet", *shlex.split(args)], cwd=CODE, check=True)
    finally:
        runs.commit()


@app.function(gpu="H100", timeout=4 * 3600, volumes=VOLUMES, secrets=KEYS)
def evaluate(args: str) -> bytes:
    out = "/tmp/result.json"
    try:
        subprocess.run(["python", "evaluate.py", *shlex.split(args), "--out", out], cwd=CODE, check=True)
    finally:
        weights.commit()  # keep the downloaded weights for the next run
    return Path(out).read_bytes()


@app.local_entrypoint()
def main(args: str, out: str, gpu: str = "H100"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(evaluate.with_options(gpu=gpu).remote(args))
    print(f"wrote {out}")

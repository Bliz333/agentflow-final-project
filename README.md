# Week 9 — Final project, part 1: AgentFlow + Flow-GRPO with LoRA (Qwen2.5-7B-Instruct, Bamboogle)

Replication of three Bamboogle rows of Table 1 of **AgentFlow** (Li et al., 2025,
[arXiv 2510.05592](https://arxiv.org/abs/2510.05592)): *Qwen-2.5-7B-Instruct*, *AgentFlow* (untrained)
and *AgentFlow w/ Flow-GRPO*, training the planner with **LoRA** instead of full fine-tuning.

| File | What it is |
|---|---|
| [`agentflow.py`](agentflow.py) | The AgentFlow loop (planner → executor → tool → verifier, then the solution generator), the official prompts, the two tools and the judge |
| [`train.py`](train.py) | Flow-GRPO: 32 questions × 8 rollouts per step, outcome reward broadcast to every planner turn, group-normalised advantage, clipped objective + KL, LoRA |
| [`evaluate.py`](evaluate.py) | Bamboogle: `direct` (the base model alone) and `agentflow` (untrained or with an adapter, 3 trials) |
| [`retriever.py`](retriever.py) | Search-R1's E5 search over the 2018 Wikipedia dump, unchanged from week 8 |
| [`test_agentflow.py`](test_agentflow.py) | Tests without a GPU model: the loop with a scripted LLM, parsers, judge, the Flow-GRPO update on a tiny random model, checkpoints |
| [`cluster/`](cluster/) | Slurm scripts for the NEU Explorer cluster (1-hour links that resume from the last step) |

## 1. Results

RESULTS_PLACEHOLDER

## 2. Setup

**As in the paper / official code** ([github.com/lupantech/AgentFlow](https://github.com/lupantech/AgentFlow)):
four modules on Qwen2.5-7B-Instruct with the official prompts (text-only branch, "direct" final
output), only the planner's next-step calls are trained, the others are frozen; Flow-GRPO with 32
questions × 8 rollouts per step, 4 updates per step, clip 0.2 / 0.3, KL 0.001 to the base model,
token-mean loss over the minibatch, planner temperature 0.7, at most 2048 planner tokens, at most 3
turns in training and 10 in evaluation, LLM-as-judge reward (1 / 0) with the official judge prompt,
3 evaluation trials.

**Different**, and why:

- **LoRA** (r = 16, α = 32, all linear layers) instead of full fine-tuning, with learning rate 1e-4
  instead of 1e-6 (LoRA needs a larger rate; 1e-5 barely moved the policy in our team's runs).
- **Tools**: `Wikipedia_Search_Tool` is Search-R1's local E5 retriever over the 2018 Wikipedia dump
  (top 3 passages) and `Base_Generator_Tool` is the base model. The official Bamboogle setup uses
  Google Search through *Gemini 2.5 Flash* with Google grounding, and a Wikipedia tool that needs
  OpenAI embeddings and gpt-4o-mini; those need paid API keys, and a local search engine keeps
  training fast and repeatable. This is the main reason the absolute numbers are lower than the paper's.
- **Judge**: the official judge prompt, but GPT-4o is replaced by DeepSeek V4.1 Flash for evaluation and
  by the local Qwen2.5-7B-Instruct for the training reward: the DeepSeek API answers only about 40
  requests a minute, which made judging 75 % of a training step. Exact match is reported next to it.
- **Training data**: Search-R1's NQ + HotpotQA questions only (the paper mixes in DeepMath, which a
  search benchmark does not need).
- Every module samples at temperature 0.7 in training and evaluation, as the official training code
  does (it passes one temperature to all modules); the paper only gives the planner's. With the
  frozen modules greedy, 78 % of the groups of 8 rollouts had identical rewards and no gradient.

## 3. How to run

The cluster layout is `/scratch/$USER/info7375-w9/{code,data,runs,results,logs}`; `data/` holds the
week 8 retrieval files (`retrieval/`, `nq_hotpotqa_train/`, `bamboogle.csv`, see week 8's README for
how they are built) and the judge key is read from `~/.config/info7375/judge.env`
(`OPENCODE_GO_API_KEY=...`). The Python environment is week 8's (torch 2.13, vLLM 0.32, PEFT 0.21).

```bash
python -m pytest -q test_agentflow.py                     # no GPU needed
sbatch -J eval cluster/eval.sbatch --mode direct --out $W/results/direct.json
sbatch -J eval cluster/eval.sbatch --mode agentflow --out $W/results/agentflow.json
sbatch -J flow_grpo cluster/train.sbatch --steps 200 --lr 1e-4   # a chain of 1-hour links
sbatch -J eval cluster/eval.sbatch --mode agentflow --adapter $W/runs/flow_grpo/adapters/step_XXXX \
  --out $W/results/flow_grpo.json
```

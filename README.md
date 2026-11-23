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
| [`diagnose.py`](diagnose.py) | Where the reward differences within a group come from (section 1) |
| [`results/`](results/) | Evaluation results (without the per-question memory, 5 MB each) and training metrics per step |
| [`test_agentflow.py`](test_agentflow.py) | Tests without a GPU model: the loop with a scripted LLM, parsers, judge, the Flow-GRPO update on a tiny random model, checkpoints |
| [`cluster/`](cluster/) | Slurm scripts for the NEU Explorer cluster (1-hour links that resume from the last step) |

## 1. Results

Bamboogle, 125 questions. **Accuracy** is the official judge prompt run by DeepSeek V4.1 Flash, mean ±
standard deviation over 3 trials (temperature 0.7, at most 10 turns); exact match (EM) in brackets.

| | paper | this replication |
|---|---|---|
| Qwen-2.5-7B-Instruct (no tools) | 12.0 | 18.4 [EM 10.4] |
| AgentFlow, untrained | 58.4 | **43.7 ± 2.0** [EM 14.7] |
| (AgentFlow, untrained, only the planner sampling) | | 45.3 ± 2.0 [EM 20.8] |
| AgentFlow w/ Flow-GRPO (LoRA), step 30 / 60 / 90 | 69.6 | 46.1 ± 1.0 / **46.9 ± 1.4** / 44.3 ± 1.6 [EM 14.9 / 17.9 / 15.5] |

- **Qwen-2.5-7B-Instruct**: the paper's 12.0 is Search-R1's exact-match number for this prompt; our
  exact match is 10.4 (13 of 125 against 15). The judge also accepts answers like "1837" for
  "20 June 1837", hence 18.4.
- **AgentFlow** is 15 points below the paper because its search tool is not the paper's: the official
  Bamboogle setup's "Google Search" is Gemini 2.5 Flash answering with Google grounding, ours is
  Search-R1's top-3 passages from the 2018 Wikipedia dump. Our teammates' independent TRL
  implementation with the same retriever gets 41–43 % (judged by Qwen2.5-7B-Instruct).
- **Flow-GRPO** did not measurably improve AgentFlow within 100 steps. Paired by question against the
  untrained system, steps 30 / 60 / 90 are +2.4 / +3.2 / +0.5 points, every 95 % bootstrap interval
  includes 0 (step 60: −2.1 to +8.8). The training reward stayed flat (about 0.59) while the planner
  moved away from the base model (KL to the base 0.45 per token at step 100), and runs with learning
  rates 1e-4 and 3e-5, which see the same questions at every step, got the same rewards (mean paired
  difference −0.004 over 37 steps).

**Why the planner does not learn here** (`diagnose.py`, `results/diagnose.json`: 64 unseen training
questions × 8 rollouts, at most 3 turns):

| | reward | groups with equal rewards | best of 8 |
|---|---|---|---|
| A every module samples at 0.7 (training default) | 0.578 | 53 % | 0.766 |
| B only the planner samples | 0.594 | 64 % | 0.734 |
| C the planner is greedy, the other modules sample | 0.574 | 53 % | 0.797 |
| D as A, planner after 30 steps at 1e-4 | 0.576 | 64 % | 0.734 |

- GRPO learns only from groups whose 8 rollouts get different rewards. With the planner greedy (C)
  as many groups differ as with everything sampled (A), more than with only the planner sampling (B):
  a large part of the reward differences the planner is credited with comes from the frozen executor,
  verifier and generator, which it cannot learn from.
- The trained planner changed its behaviour (5.1 instead of 3.6 distinct first tool calls per group)
  but not the reward on these questions (D − A = −0.002 ± 0.020 per question).
- The room is small: best of 8 is 0.77 against a mean of 0.58, and the planner picks the Wikipedia
  search for 98 % of its tool calls. Choosing a tool, which in the paper's setup decides between
  Wikipedia and the Gemini-backed Google Search, leaves it little to improve here.

## 2. Setup

**As in the paper / official code** ([github.com/lupantech/AgentFlow](https://github.com/lupantech/AgentFlow)):
four modules on Qwen2.5-7B-Instruct with the official prompts (text-only branch, "direct" final
output), only the planner's next-step calls are trained, the others are frozen; Flow-GRPO with 32
questions × 8 rollouts per step, 4 updates per step, clip 0.2 / 0.3, KL 0.001 to the base model,
token-mean loss over the minibatch, temperature 0.7, at most 2048 planner tokens, at most 3 turns in
training and 10 in evaluation, the official judge prompt, 3 evaluation trials.

**Different**, and why:

- **LoRA** (r = 16, α = 32, all linear layers) instead of full fine-tuning, with learning rate 1e-4
  instead of 1e-6 (LoRA needs a larger rate; 1e-5 barely moved the policy in our team's runs).
- **Tools**: `Wikipedia_Search_Tool` is Search-R1's local E5 retriever over the 2018 Wikipedia dump
  (top 3 passages) and `Base_Generator_Tool` is the base model. The official Bamboogle setup uses
  Google Search through *Gemini 2.5 Flash* with Google grounding, and a Wikipedia tool that needs
  OpenAI embeddings and gpt-4o-mini; those need paid API keys, and a local search engine keeps
  training fast and repeatable. This is the main reason the absolute numbers are lower than the paper's.
- **Judge**: evaluation uses the official judge prompt with DeepSeek V4.1 Flash instead of GPT-4o;
  exact match is reported next to it. The **training reward** is not an LLM judge but FlashRAG's
  accuracy rule (a golden answer appears in the final answer): the DeepSeek API answers only about 40
  requests a minute, which made judging 75 % of a training step, and the local Qwen2.5-7B-Instruct as
  judge rejected 39 % of the answers DeepSeek accepts. On 500 Bamboogle answers the rule agrees with
  DeepSeek 92 % of the time (Qwen as judge: 86 %, exact match: 80 %).
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
python diagnose.py --adapter $W/runs/flow_grpo/adapters/step_0030 --out $W/results/diagnose.json   # one GPU
```

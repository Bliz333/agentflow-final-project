# AgentFlow + Flow-GRPO with LoRA (Qwen2.5-7B-Instruct) — INFO 7375 final project

**Team**: Tianyue Gao, Shizhao Han, Shancheng Mi, Hoi Shan Tang. Team repository:
[Bliz333/agentflow-final-project](https://github.com/Bliz333/agentflow-final-project) (public; submitted on
Canvas by its link).

## Part 1 — Bamboogle

Replication of three Bamboogle rows of Table 1 of **AgentFlow** (Li et al., 2025,
[arXiv 2510.05592](https://arxiv.org/abs/2510.05592)): *Qwen-2.5-7B-Instruct*, *AgentFlow* (untrained)
and *AgentFlow w/ Flow-GRPO*, training the planner with **LoRA** instead of full fine-tuning.

| File | What it is |
|---|---|
| [`agentflow.py`](agentflow.py) | The AgentFlow loop (planner → executor → tool → verifier, then the solution generator), the official prompts, the tools and the judge |
| [`train.py`](train.py) | Flow-GRPO: 32 questions × 8 rollouts per step, outcome reward broadcast to every planner turn, group-normalised advantage, clipped objective + KL, LoRA |
| [`evaluate.py`](evaluate.py) | Bamboogle: `direct` (the base model alone) and `agentflow` (untrained or with an adapter, 3 trials) |
| [`retriever.py`](retriever.py) | Search-R1's E5 search over the 2018 Wikipedia dump (from our Search-R1 replication) |
| [`prep_index.py`](prep_index.py) | Builds `data/retrieval` from the files Search-R1 publishes (run once) |
| [`wikipedia.py`](wikipedia.py) | The official Wikipedia_Search_Tool on live Wikipedia (`--search wikipedia`, section 1) |
| [`google_search.py`](google_search.py) | The official Google_Search_Tool with DeepSeek's web search in place of Gemini (`--google-search`, section 1) |
| [`diagnose.py`](diagnose.py) | Where the reward differences within a group come from (section 1) |
| [`modal_app.py`](modal_app.py) | Runs `evaluate.py` and `train.py` on one Modal GPU, and prepares their data there |
| [`results/`](results/) | Evaluation results (without the per-question memory, 5 MB each) and training metrics per step |
| [`test_agentflow.py`](test_agentflow.py) | Tests without a GPU model: the loop with a scripted LLM, parsers, judge, the Flow-GRPO update on a tiny random model, checkpoints |
| [`cluster/`](cluster/) | Slurm scripts for the NEU Explorer cluster (1-hour links that resume from the last step) |

## 1. Results

Bamboogle, 125 questions. **Accuracy** is the official judge prompt run by DeepSeek V4.1 Flash, mean ±
standard deviation over 3 trials (temperature 0.7, at most 10 turns); exact match (EM) in brackets.

With the paper's tools except `Python_Coder_Tool` (`Base_Generator_Tool`, `Google_Search_Tool`,
`Wikipedia_Search_Tool`), in training and evaluation alike. Google Search is DeepSeek's web search in place of Gemini 2.5 Flash with Google
grounding, which is retired (end of this section); Wikipedia is Search-R1's local retriever over the 2018
dump:

| | paper | this replication |
|---|---|---|
| Qwen-2.5-7B-Instruct (no tools) | 12.0 | 18.4 [EM 10.4] |
| AgentFlow, untrained | 58.4 | **68.0 ± 1.7** [EM 22.1] |
| AgentFlow w/ Flow-GRPO (LoRA), step 30 | 69.6 | **71.2 ± 3.0** [EM 26.1] |

- **Qwen-2.5-7B-Instruct**: the paper's 12.0 is Search-R1's exact-match number for this prompt; our
  exact match is 10.4 (13 of 125 against 15). The judge also accepts answers like "1837" for
  "20 June 1837", hence 18.4.
- **AgentFlow** is 9.6 points above the paper (`results/agentflow_google_local.json`); possible reasons, not
  tested: DeepSeek V4.1 Flash with its web search may be a stronger tool than Gemini 2.5 Flash was, and
  the judge is DeepSeek instead of GPT-4o.
- **Flow-GRPO** (`results/flow_grpo_google_step_0030.json`, run `flow_grpo_google`): +3.2 points paired by
  question (95 % bootstrap interval −1.9 to +8.3; 35 questions better, 22 worse), EM +4.0. The paper
  gains 11.2 from a lower start. The training reward stayed level (0.66 / 0.67 / 0.64 over steps 1–10 /
  11–20 / 21–30) while the planner moved from the base model (KL 0.15 per token at step 30); the trained
  planner calls Wikipedia more (505 instead of 286 calls over the 3 trials) and takes more turns (2.4
  instead of 1.9). Training stopped at 30 steps because of the search cost (steps 21–30 and the step-30
  evaluation together cost ¥8.2 of DeepSeek tokens at off-peak prices). Steps 1–20 ran on Explorer, 21–30
  on Modal (both one H200, same data). From step 20 on, rollouts of a step that send the same query share
  one search (steps 1–19 searched every query; the untrained evaluation too), to save tokens.

**Without the Google Search tool** (our first setup: `Base_Generator_Tool` and the local
`Wikipedia_Search_Tool` only):

| | paper | this replication |
|---|---|---|
| AgentFlow, untrained | 58.4 | 43.7 ± 2.0 [EM 14.7] |
| (AgentFlow, untrained, only the planner sampling) | | 45.3 ± 2.0 [EM 20.8] |
| AgentFlow w/ Flow-GRPO (LoRA), step 30 / 60 / 90 | 69.6 | 46.1 ± 1.0 / 46.9 ± 1.4 / 44.3 ± 1.6 [EM 14.9 / 17.9 / 15.5] |
| (Flow-GRPO with the frozen modules greedy in training, step 60) | | 44.5 ± 3.1 [EM 16.8] |
| (same with 96 questions per step, step 30 / 60) | | 46.7 ± 2.6 / 45.3 ± 1.4 [EM 17.6 / 15.5] |

- **AgentFlow** is then 15 points below the paper because its search tool is not the paper's: the official
  Bamboogle setup's "Google Search" is Gemini 2.5 Flash answering with Google grounding, ours is
  Search-R1's top-3 passages from the 2018 Wikipedia dump. Our teammates' independent TRL
  implementation with the same retriever gets 41–43 % (judged by Qwen2.5-7B-Instruct). With live
  Wikipedia and a Google Search tool (DeepSeek's web search standing in for Gemini, end of this section)
  untrained AgentFlow gets 65.1, 6.7 points above the paper.
- **Flow-GRPO** did not measurably improve AgentFlow within 100 steps. Paired by question against the
  untrained system, steps 30 / 60 / 90 are +2.4 / +3.2 / +0.5 points, every 95 % bootstrap interval
  includes 0 (step 60: −2.1 to +8.8). The training reward stayed flat (about 0.59) while the planner
  moved away from the base model (KL to the base 0.45 per token at step 100), and runs with learning
  rates 1e-4 and 3e-5, which see the same questions at every step, got the same rewards (mean paired
  difference −0.004 over 37 steps). Training with the frozen modules greedy, so that the reward
  differences within a group come from the planner alone (below), did not help either: +0.8 at step 60,
  and with 96 questions per step (about 24 groups with a training signal per step instead of 8) +2.9 / +1.6
  at steps 30 / 60, again with intervals that include 0.

| Run (`results/metrics/`) | Setting | Steps |
|---|---|---|
| `flow_grpo_google` | the main run, with Google Search: every module at 0.7, learning rate 1e-4 | 30 |
| `flow_grpo_lr1e-4` | without Google Search: every module at 0.7, learning rate 1e-4 | 101 |
| `flow_grpo_lr3e-5` | learning rate 3e-5, stopped when its rewards matched the 1e-4 run | 37 |
| `flow_grpo_lr1e-4_greedy` | frozen modules greedy (`--module-temperature 0`) | 75 |
| `flow_grpo_greedy_q96` | frozen modules greedy, 96 questions per step | 72 |

The first three runs used the reward rule on substrings, the last one whole words, as `train.py` now does (a
golden "8" no longer matches "1988"), which changes the reward only for the 3.4 % of training
questions with a golden answer of at most two characters.

**Why the planner does not learn here** (`diagnose.py`, `results/diagnose.json`: 64 unseen training
questions × 8 rollouts, at most 3 turns; run while the reward still matched substrings, not whole words):

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

**Is a better Wikipedia tool the missing piece? No.** `wikipedia.py` implements the official
`Wikipedia_Search_Tool` as its code intends: Wikipedia's search (10 titles), the LLM picks up to 3 pages
from the titles, each page is cut into 200-word chunks, the 10 chunks closest to the query (E5) are kept,
and the LLM answers from them, all with the official prompts and Qwen2.5-7B-Instruct as the tool's LLM.
(In the released official code the per-page step never runs: the tool's `model_string` is never set,
creating the page tool raises, and the tool returns only titles with 100-character abstracts.) Untrained
AgentFlow with it, on live Wikipedia (`results/agentflow_live_wikipedia.json`):

| Untrained AgentFlow, Wikipedia_Search_Tool | Bamboogle accuracy | EM | turns |
|---|---|---|---|
| Search-R1's local retriever (2018 dump, top 3 passages) | 43.7 ± 2.0 | 14.7 | 5.5 |
| Official tool on live Wikipedia | 44.3 ± 1.9 (45.6 / 41.6 / 45.6) | 16.8 | 2.6 |

The two tools are equally accurate (paired by question +0.5 points, 95 % interval −6.4 to +7.5). The
official tool's synthesized answer lets the verifier stop earlier (25 instead of 107 of the 375 runs use
all 10 turns), but it does not answer more questions; 5 % of its searches find nothing, nearly all
queries written as whole questions. (Wikimedia refuses bursts of requests with HTTP 429; a first run that
counted refused requests as empty searches scored 39.5 and is discarded.) So the gap to the paper's 58.4
is not the Wikipedia tool but the paper's Google Search tool, Gemini 2.5 Flash answering with Google
grounding, a much stronger model than any module here.

**With the paper's Google Search tool: 65.1.** `google_search.py` is the official `Google_Search_Tool`,
which sends the query unchanged to Gemini 2.5 Flash with Google Search grounding and returns its answer
with "[n](url)" citations. Gemini 2.5 Flash is closed to new API users and retired on 1 December 2026, so
DeepSeek's server-side web search stands in for it (`deepseek-flash`, i.e. DeepSeek-V4.1-Flash, thinking
off; the query goes to it unchanged, the citations are added the same way, and the planner sees the
official tool description). Untrained AgentFlow with both search tools
(`--search wikipedia --google-search`, `results/agentflow_google_search.json`):

| Untrained AgentFlow | Bamboogle accuracy | EM | turns |
|---|---|---|---|
| Wikipedia_Search_Tool only (live Wikipedia, above) | 44.3 ± 1.9 | 16.8 | 2.6 |
| + Google_Search_Tool (DeepSeek web search) | **65.1 ± 2.9** (61.6 / 68.8 / 64.8) | 19.2 | 1.6 |

- Paired by question +20.8 points (95 % interval +13.6 to +28.0; 60 questions better, 14 worse), and
  +18.1 over the best Flow-GRPO adapter (step 60, trained without this tool).
- The planner calls Google Search for 59 % of its tool calls (Wikipedia 40 %). 283 of the 375 runs use
  it and are right 70 % of the time, against 49 % for the runs that do not.
- Leakage: 3 of the 359 Google Search results cite a page that contains Bamboogle's answers (a GitHub
  commit, a copy of the dataset on Hugging Face); counting those 3 runs as wrong gives 64.3.
- So the paper's numbers rest mostly on this tool: a strong model searching the web answers most of
  Bamboogle's two-hop questions in one call. Without it the planner had little to choose (98 % of its
  calls went to Wikipedia); with it, which tool to call and what to ask become real choices, the ones
  Flow-GRPO trains. Flow-GRPO with this tool: the first table of this section.

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
  `--google-search` adds `Google_Search_Tool` with DeepSeek's web search in place of Gemini 2.5 Flash,
  which is retired; the main results train and evaluate with it. The official tool set's
  `Python_Coder_Tool` is left out: Bamboogle's questions are fact lookups.
- **Judge**: evaluation uses the official judge prompt with DeepSeek V4.1 Flash instead of GPT-4o;
  exact match is reported next to it. The **training reward** is not an LLM judge but FlashRAG's
  accuracy rule on whole words (a golden answer appears in the final answer): the DeepSeek API answers only about 40
  requests a minute, which made judging 75 % of a training step, and the local Qwen2.5-7B-Instruct as
  judge rejected 39 % of the answers DeepSeek accepts. On 500 Bamboogle answers the rule agrees with
  DeepSeek 92 % of the time (Qwen as judge: 86 %, exact match: 80 %).
- **Training data**: Search-R1's NQ + HotpotQA questions only (the paper mixes in DeepMath, which a
  search benchmark does not need).
- Every module samples at temperature 0.7 in training and evaluation, as the official training code
  does (it passes one temperature to all modules); the paper only gives the planner's.

## 3. How to run

Tests need no GPU: `python -m pytest -q test_agentflow.py`. Training and evaluation need one GPU with
141 GB (H200) for training, 80 GB for evaluation, and about 110 GB of disk for the retrieval data.

**Environment** (cluster nodes have CUDA 12.x drivers, hence the `cu129` builds):

```bash
uv venv --seed --python 3.12 /scratch/$USER/envs/agentflow   # the path cluster/env.sh uses
uv pip install --python /scratch/$USER/envs/agentflow/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" "torchaudio==2.11.0+cu129" \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  -r requirements.txt
```

**Data** (in `data/` next to the code):

```bash
hf download --repo-type dataset PeterJinGo/nq_hotpotqa_train --local-dir data/nq_hotpotqa_train
hf download --repo-type dataset PeterJinGo/wiki-18-corpus --local-dir data/wiki-18-corpus
hf download --repo-type dataset PeterJinGo/wiki-18-e5-index --local-dir data/wiki-18-e5-index
curl -L -o data/bamboogle.csv "https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U/export?format=csv&gid=0"
python prep_index.py --index-dir data/wiki-18-e5-index --corpus data/wiki-18-corpus/wiki-18.jsonl.gz --out data/retrieval
export HF_HOME=/scratch/$USER/hf-home    # the jobs run with HF_HUB_OFFLINE=1 and read the models from here
hf download Qwen/Qwen2.5-7B-Instruct
hf download intfloat/e5-base-v2
```

**Judge**: an OpenCode Go key in `~/.config/info7375/judge.env` as `OPENCODE_GO_API_KEY=...` (any
OpenAI-compatible API works by changing `JUDGE_URL` / `JUDGE_MODEL` in `agentflow.py`).
`--google-search` needs a DeepSeek key (platform.deepseek.com) in `~/.config/info7375/deepseek.env` as
`DEEPSEEK_API_KEY=...`; OpenCode Go does not pass DeepSeek's web search through.

**On the NEU Explorer cluster** the scripts expect `/scratch/$USER/info7375-w9/{code,data,runs,results,logs}`
(code = this repository, `code/data` a link to `../data`); set your Slurm account (`-A`) in
`cluster/*.sbatch`.

On Modal (an H100 for evaluation, an H200 for training; the 16-minute live-Wikipedia evaluation cost
about $1.1). The judge key, and for `--google-search` the DeepSeek key, must be in the environment, e.g.
`set -a; . ~/.config/info7375/judge.env; . ~/.config/info7375/deepseek.env; set +a`. Google Search costs
about 5k DeepSeek tokens per search (about ¥5 per 1,000 off-peak, twice that in Beijing's working hours;
in training, a step makes about 260 calls, fewer after repeated queries are merged); DeepSeek lowers the
allowed concurrency when the balance is low:

```bash
modal run modal_app.py::check                     # CPU only: build the image, run the tests
modal run modal_app.py::prep                      # CPU only, once (10 minutes): retrieval data, training questions
modal run modal_app.py --args "--mode agentflow --search wikipedia --vllm-mem 0.85" --out results/agentflow_live_wikipedia.json
modal run modal_app.py --args "--mode agentflow --search wikipedia --google-search" --out results/agentflow_google_search.json
modal run modal_app.py::train --args "--out /runs/flow_grpo_google --resume --steps 30 --google-search"
modal run modal_app.py --args "--mode agentflow --google-search --retrieval /data/retrieval \
  --adapter /runs/flow_grpo_google/adapters/step_0030" --out results/flow_grpo_google_step_0030.json
```

```bash
W=/scratch/$USER/info7375-w9
sbatch -J eval cluster/eval.sbatch --mode direct --out $W/results/direct.json
sbatch -J eval cluster/eval.sbatch --mode agentflow --out $W/results/agentflow.json
sbatch -J flow_grpo cluster/train.sbatch --steps 200 --lr 1e-4   # a chain of 1-hour links
sbatch -J flow_grpo_google cluster/train.sbatch --steps 30 --lr 1e-4 --google-search
sbatch -J flow_grpo_greedy cluster/train.sbatch --steps 200 --lr 1e-4 --module-temperature 0
sbatch -J eval cluster/eval.sbatch --mode agentflow --adapter $W/runs/flow_grpo/adapters/step_XXXX \
  --out $W/results/flow_grpo.json
python diagnose.py --adapter $W/runs/flow_grpo/adapters/step_0030 --out $W/results/diagnose.json   # one GPU
```

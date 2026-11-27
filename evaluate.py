"""AgentFlow's Table 1 rows for Qwen2.5-7B-Instruct on Bamboogle (125 questions, Part 1), on KGQAGen-10k's test
split (Part 2: --dataset kgqagen) or on LiveSearchBench's 2025 set (300 questions: --dataset livesearchbench).

  direct     the base model answers alone, greedy, with Search-R1's Direct Inference prompt (the paper's
             Qwen-2.5-7B-Instruct row, 12.0, is Search-R1's number for this setting)
  agentflow  AgentFlow, untrained or with a Flow-GRPO adapter: at most 10 turns, every module at temperature
             0.7 (the paper's planner temperature; the official code uses one temperature), 3 trials

Accuracy is the judge's verdict (agentflow.judge); exact match of the <answer> is reported next to it.
    python evaluate.py --mode direct --out results/direct.json
    python evaluate.py --mode agentflow --out results/agentflow.json
    python evaluate.py --mode agentflow --adapter runs/flow_grpo/adapters/step_0100 --out results/flow_grpo.json
"""
import argparse
import csv
import io
import json
import os
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import agentflow
from google_search import DeepSeekSearch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BAMBOOGLE_CSV = ("https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U"
                 "/export?format=csv&gid=0")
LIVESEARCHBENCH = ("https://raw.githubusercontent.com/hengzzzhou/LiveSearchbench/"
                   "e0286dc1361bbd91e663a7423f4b3f8959c14b15/bench/2025/level{}.json")
KGQAGEN = ("https://huggingface.co/datasets/lianglz/KGQAGen-10k/resolve/145034b656c0009568697e370afe6a18f6c7c053/"
           "data/{}-00000-of-00001.parquet")
# FlashRAG's zero-shot prompt, which Search-R1 uses for Direct Inference.
DIRECT_SYSTEM = ("Answer the question based on your own knowledge. "
                 "Only give me the answer and do not output any other words.")


def load_kgqagen(split, cache):
    """KGQAGen-10k (Zhang et al., 2025, arXiv 2505.23495, NeurIPS 2025 Datasets and Benchmarks): multi-hop questions
    generated from Wikidata, split into train 8,629 / dev 1,079 / test 1,079; each answer is a list of entity names."""
    import pandas as pd
    cache = Path(cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(KGQAGEN.format(split)).read())
    data = pd.read_parquet(cache)
    return [{"question": q.strip(), "golden_answers": [a.strip() for a in g]}
            for q, g in zip(data["question"], data["answer"])]


def load_livesearchbench(cache):
    """LiveSearchBench (Zhou et al., 2025, arXiv 2511.01409), its 2025 set: levels 1-3, 150 + 100 + 50 questions
    built from Wikidata changes, one answer each (a Wikidata label), at the repository's commit of 1 Sep 2026."""
    cache = Path(cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([qa for level in (1, 2, 3) for qa in
                                     json.load(urllib.request.urlopen(LIVESEARCHBENCH.format(level)))["qa_pairs"]]))
    return [{"question": x["question"].strip(), "golden_answers": [x["answer"].strip()]}
            for x in json.loads(cache.read_text())]


def load_bamboogle(cache):
    """The Google Sheet linked in the assignment (FlashRAG's 125 questions, in the same order)."""
    cache = Path(cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(BAMBOOGLE_CSV).read())
    rows = list(csv.reader(io.StringIO(cache.read_text(encoding="utf-8"))))[1:]
    return [{"question": r[0].strip(), "golden_answers": [a.strip() for a in r[4:6] if a.strip()]}
            for r in rows if r and r[0].strip()]


class ApiLLM:
    """vLLM's LLM.generate through an OpenAI-compatible API, for the base model only (no LoRA); the prompts are
    chat messages (see Messages)."""

    def __init__(self, base_url, model, key_env, workers=16):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ[key_env], timeout=300, max_retries=20)
        self.model, self.workers = model, workers

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        assert lora_request is None, "the API serves the base model, not a LoRA adapter"

        def one(messages):
            reply = self.client.chat.completions.create(model=self.model, messages=messages,
                                                        temperature=params.temperature, max_tokens=params.max_tokens)
            text = reply.choices[0].message.content or ""
            return SimpleNamespace(prompt_token_ids=[], outputs=[SimpleNamespace(text=text, token_ids=[], logprobs=None)])

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(one, prompts))


class Messages:
    """The tokenizer for ApiLLM: the API applies the chat template, so prompts stay chat messages."""

    @staticmethod
    def apply_chat_template(messages, **kwargs):
        return messages


def search_tool(args):
    if args.search == "wikipedia":
        from wikipedia import LiveWikipedia
        return LiveWikipedia()
    from retriever import Retriever
    return agentflow.LocalWikipedia(Retriever(args.retrieval))  # 33 GB on the GPU


def score(data, outputs):
    """One row per question: the answer, the judge's verdict and exact match."""
    answers = [agentflow.final_answer(o) for o in outputs]
    verdicts = agentflow.judge([x["question"] for x in data], answers, [x["golden_answers"] for x in data])
    return [{"question": x["question"], "golden_answers": x["golden_answers"], "output": o, "answer": a,
             "correct": v, "em": agentflow.exact_match(a, x["golden_answers"])}
            for x, o, a, v in zip(data, outputs, answers, verdicts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["direct", "agentflow"], required=True)
    parser.add_argument("--adapter", help="Flow-GRPO LoRA adapter directory (none = untrained)")
    parser.add_argument("--planner-model", help="a full model for the planner's next-step calls, e.g. the paper's "
                                                "released AgentFlow/agentflow-planner-7b; the other modules stay on --model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--search", choices=["local", "wikipedia"], default="local",
                        help="Wikipedia_Search_Tool: Search-R1's local retriever or the official tool on live Wikipedia")
    parser.add_argument("--google-search", action="store_true",
                        help="add the official Google_Search_Tool, answered by DeepSeek's web search (DEEPSEEK_API_KEY)")
    parser.add_argument("--retrieval", default="data/retrieval")
    parser.add_argument("--dataset", choices=["bamboogle", "kgqagen", "livesearchbench"], default="bamboogle")
    parser.add_argument("--bamboogle", default="data/bamboogle.csv")
    parser.add_argument("--livesearchbench", default="data/livesearchbench_2025.json")
    parser.add_argument("--kgqagen", default="data/kgqagen_test.parquet")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    parser.add_argument("--model", default=MODEL_ID, help="a smaller model for local smoke tests")
    parser.add_argument("--vllm-mem", type=float, help="vLLM's share of the GPU (default 0.85 direct, 0.45 agentflow)")
    parser.add_argument("--api-base", help="run the base model through this OpenAI-compatible API instead of vLLM")
    parser.add_argument("--api-model", default=MODEL_ID, help="the base model's name at --api-base")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY", help="environment variable with the --api-base key")
    args = parser.parse_args()
    if args.api_base and (args.adapter or args.planner_model):
        parser.error("--api-base serves the base model only; evaluate a trained planner with vLLM")
    if args.adapter and args.planner_model:
        parser.error("--adapter and --planner-model are two different trained planners")
    os.environ["OPENCODE_GO_API_KEY"]  # the judge's key: fail now, not after the GPU rollouts

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    data = {"bamboogle": lambda: load_bamboogle(args.bamboogle), "kgqagen": lambda: load_kgqagen("test", args.kgqagen),
            "livesearchbench": lambda: load_livesearchbench(args.livesearchbench)}[args.dataset]()[:args.limit]
    api = ApiLLM(args.api_base, args.api_model, args.api_key_env) if args.api_base else None
    tokenizer = Messages() if api else AutoTokenizer.from_pretrained(args.model)
    trials = []
    if args.mode == "direct":
        llm = api or LLM(args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.vllm_mem or 0.85,
                         seed=0)
        prompts = [tokenizer.apply_chat_template([{"role": "system", "content": DIRECT_SYSTEM},
                                                  {"role": "user", "content": f"Question: {x['question']}"}],
                                                 tokenize=False, add_generation_prompt=True) for x in data]
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512), use_tqdm=False)
        trials.append(score(data, [o.outputs[0].text for o in outputs]))
    else:
        wikipedia = search_tool(args)  # before vLLM, which then takes its share of the GPU
        google = DeepSeekSearch() if args.google_search else None
        llm = api or LLM(args.model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=args.vllm_mem or 0.45,
                         seed=0, enable_lora=bool(args.adapter), max_lora_rank=16)
        lora = LoRARequest("adapter", 1, str(args.adapter)) if args.adapter else None
        planner = LLM(args.planner_model, dtype="bfloat16", max_model_len=32768,
                      gpu_memory_utilization=args.vllm_mem or 0.45, seed=0) if args.planner_model else None
        questions = [f"{x['question']} {agentflow.ANSWER_FORMAT}" for x in data]
        for trial in range(args.trials):
            episodes = agentflow.solve(llm, tokenizer, wikipedia, questions, args.max_steps, args.temperature, lora,
                                       google=google, planner_llm=planner)
            rows = score(data, [ep.output for ep in episodes])
            for row, ep in zip(rows, episodes):
                row["memory"] = ep.memory
            trials.append(rows)
            print(f"trial {trial}: accuracy {sum(r['correct'] for r in rows) / len(rows):.3f}", flush=True)
    accuracy = [sum(r["correct"] for r in rows) / len(rows) for rows in trials]
    summary = {"dataset": args.dataset, "mode": args.mode, "adapter": args.adapter or args.planner_model, "n": len(data), "accuracy": statistics.mean(accuracy),
               "accuracy_std": statistics.pstdev(accuracy), "trials": accuracy,
               "em": statistics.mean(sum(r["em"] for r in rows) / len(rows) for rows in trials)}
    if args.mode == "agentflow":
        summary["turns"] = statistics.mean(len(r["memory"]) for rows in trials for r in rows)
    print(json.dumps(summary), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "trials": trials}, indent=1))


if __name__ == "__main__":
    main()

"""Bamboogle (125 questions): AgentFlow's Table 1 rows for Qwen2.5-7B-Instruct.

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
from pathlib import Path

import agentflow

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BAMBOOGLE_CSV = ("https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U"
                 "/export?format=csv&gid=0")
# FlashRAG's zero-shot prompt, which Search-R1 uses for Direct Inference.
DIRECT_SYSTEM = ("Answer the question based on your own knowledge. "
                 "Only give me the answer and do not output any other words.")


def load_bamboogle(cache):
    """The Google Sheet linked in the assignment (FlashRAG's 125 questions, in the same order)."""
    cache = Path(cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(BAMBOOGLE_CSV).read())
    rows = list(csv.reader(io.StringIO(cache.read_text(encoding="utf-8"))))[1:]
    return [{"question": r[0].strip(), "golden_answers": [a.strip() for a in r[4:6] if a.strip()]}
            for r in rows if r and r[0].strip()]


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
    parser.add_argument("--out", required=True)
    parser.add_argument("--retrieval", default="data/retrieval")
    parser.add_argument("--bamboogle", default="data/bamboogle.csv")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    parser.add_argument("--model", default=MODEL_ID, help="a smaller model for local smoke tests")
    parser.add_argument("--vllm-mem", type=float, help="vLLM's share of the GPU (default 0.85 direct, 0.45 agentflow)")
    args = parser.parse_args()
    os.environ["OPENCODE_GO_API_KEY"]  # the judge's key: fail now, not after the GPU rollouts

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    data = load_bamboogle(args.bamboogle)[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    trials = []
    if args.mode == "direct":
        llm = LLM(args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.vllm_mem or 0.85,
                  seed=0)
        prompts = [tokenizer.apply_chat_template([{"role": "system", "content": DIRECT_SYSTEM},
                                                  {"role": "user", "content": f"Question: {x['question']}"}],
                                                 tokenize=False, add_generation_prompt=True) for x in data]
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512), use_tqdm=False)
        trials.append(score(data, [o.outputs[0].text for o in outputs]))
    else:
        from retriever import Retriever
        retriever = Retriever(args.retrieval)  # 33 GB on the GPU, before vLLM takes its share
        llm = LLM(args.model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=args.vllm_mem or 0.45,
                  seed=0, enable_lora=bool(args.adapter), max_lora_rank=16)
        lora = LoRARequest("adapter", 1, str(args.adapter)) if args.adapter else None
        questions = [f"{x['question']} {agentflow.ANSWER_FORMAT}" for x in data]
        for trial in range(args.trials):
            episodes = agentflow.solve(llm, tokenizer, retriever, questions, args.max_steps, args.temperature, lora)
            rows = score(data, [ep.output for ep in episodes])
            for row, ep in zip(rows, episodes):
                row["memory"] = ep.memory
            trials.append(rows)
            print(f"trial {trial}: accuracy {sum(r['correct'] for r in rows) / len(rows):.3f}", flush=True)
    accuracy = [sum(r["correct"] for r in rows) / len(rows) for rows in trials]
    summary = {"mode": args.mode, "adapter": args.adapter, "n": len(data), "accuracy": statistics.mean(accuracy),
               "accuracy_std": statistics.pstdev(accuracy), "trials": accuracy,
               "em": statistics.mean(sum(r["em"] for r in rows) / len(rows) for rows in trials)}
    if args.mode == "agentflow":
        summary["turns"] = statistics.mean(len(r["memory"]) for rows in trials for r in rows)
    print(json.dumps(summary), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "trials": trials}, indent=1))


if __name__ == "__main__":
    main()

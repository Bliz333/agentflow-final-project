"""Where do the reward differences within a group of 8 rollouts come from, and does training change reward?

64 unseen training questions x 8 rollouts (at most 3 turns, as in training) under four conditions:
  A  planner 0.7, other modules 0.7      (what train.py samples by default)
  B  planner 0.7, other modules greedy   (differences come from the planner only)
  C  planner greedy, other modules 0.7   (differences come from the frozen modules only)
  D  as A, with a trained planner adapter
    python diagnose.py --adapter runs/flow_grpo_lr1e-4/adapters/step_0030 --out results/diagnose.json
"""
import argparse
import json
import statistics

import agentflow
import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/nq_hotpotqa_train/train.parquet")
    parser.add_argument("--retrieval", default="data/retrieval")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from retriever import Retriever

    tokenizer = AutoTokenizer.from_pretrained(train.MODEL_ID)
    retriever = Retriever(args.retrieval)
    llm = LLM(train.MODEL_ID, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.45, seed=1,
              enable_lora=True, max_lora_rank=16)
    questions = train.load_questions(args.data, 0)[20000:20064]  # far past what any run has trained on
    prompts = [f"{q.strip()} {agentflow.ANSWER_FORMAT}" for q, _ in questions for _ in range(8)]
    golden = [g for _, g in questions for _ in range(8)]
    adapter = LoRARequest("adapter", 1, args.adapter)
    out = {}
    for name, planner, others, lora in [("A", .7, .7, None), ("B", .7, 0, None), ("C", 0, .7, None),
                                        ("D", .7, .7, adapter)]:
        episodes = agentflow.solve(llm, tokenizer, retriever, prompts, 3, planner, lora, module_temperature=others)
        rewards = [agentflow.cover_match(agentflow.final_answer(ep.output), g) for ep, g in zip(episodes, golden)]
        groups = [rewards[i:i + 8] for i in range(0, len(rewards), 8)]
        firsts = [[ep.memory.get("Action Step 1", {}) for ep in episodes[i:i + 8]] for i in range(0, len(episodes), 8)]
        out[name] = {"reward": statistics.mean(rewards),
                     "zero_std": statistics.mean(len(set(g)) == 1 for g in groups),
                     "pass8": statistics.mean(max(g) for g in groups),
                     "turns": statistics.mean(len(ep.memory) for ep in episodes),
                     "distinct_first_subgoals": statistics.mean(len({m.get("sub_goal") for m in f}) for f in firsts),
                     "distinct_first_commands": statistics.mean(len({m.get("command") for m in f}) for f in firsts),
                     "per_question": [statistics.mean(g) for g in groups]}
        print(name, {k: v for k, v in out[name].items() if k != "per_question"}, flush=True)
    diff = [d - a for a, d in zip(out["A"]["per_question"], out["D"]["per_question"])]
    print("D - A per question: mean %.3f, standard error %.3f" % (statistics.mean(diff), statistics.stdev(diff) / 8))
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()

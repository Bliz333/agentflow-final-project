"""Flow-GRPO (AgentFlow, Eq. 5) with LoRA: train the planner of agentflow.py on Search-R1's NQ + HotpotQA.

Each step: 32 questions x 8 AgentFlow rollouts (at most 3 turns, temperature 0.7); a final answer gets
reward 1 if a golden answer appears in it (agentflow.cover_match), else 0. The paper's GPT-4o judge is
replaced by this rule: on 500 Bamboogle answers it agrees with the DeepSeek judge used for evaluation 92 %
of the time, the local Qwen2.5-7B-Instruct as judge only 86 % (it rejects correct answers written as a
sentence), and the DeepSeek API is too slow for training (about 40 requests a minute). A rollout's reward, normalised within its group of 8 (Eq. 7), is the
advantage of every planner turn in it (Eq. 4: the final outcome is broadcast to all turns). Then 4
minibatch updates of the clipped objective (clip 0.2 / 0.3) plus a KL penalty to the base model (k3,
beta 0.001), averaged over all planner tokens of the minibatch, as the official verl code does.

The old log-probs are vLLM's sampling log-probs (after temperature). vLLM gets the current LoRA as a
LoRA adapter every step, not merged into the bf16 weights (merging rounds most small updates away).
A checkpoint (adapter + optimizer) is written after every step; --resume continues from the last one.
    python train.py --out runs/flow_grpo --lr 1e-4
"""
import argparse
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import agentflow
from evaluate import search_tool

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_questions(path, seed):
    import pandas as pd
    data = pd.read_parquet(path).sample(frac=1.0, random_state=seed)
    return [(q, list(g)) for q, g in zip(data["question"], data["golden_answers"])]


def token_logprobs(model, prompt_ids, completion_ids, temperature):
    """log pi(completion token | prefix) under the model, with the sampling temperature applied."""
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    logits = model(input_ids=ids, logits_to_keep=len(completion_ids) + 1).logits[0, :-1].float() / temperature
    return logits.log_softmax(-1).gather(1, ids[0, len(prompt_ids):, None]).squeeze(1)


def last_checkpoint(out):
    done = sorted(p.parent for p in out.glob("checkpoint_*/state.pt"))  # state.pt is written last
    return done[-1] if done else None


def save_checkpoint(model, optimizer, out, step):
    path = out / f"checkpoint_{step:04d}"
    model.save_pretrained(path)
    # state.pt marks a complete checkpoint, so it appears only once fully written (a link can be killed).
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, path / "state.tmp")
    os.replace(path / "state.tmp", path / "state.pt")
    for old in out.glob("checkpoint_*"):
        if old != path:
            shutil.rmtree(old)


def policy_loss(model, samples, n_tokens, args):
    """Backpropagate one minibatch: samples are (prompt_ids, completion_ids, old_logprobs, advantage)."""
    stats = Counter()
    for prompt_ids, completion_ids, old, advantage in samples:
        logp = token_logprobs(model, prompt_ids, completion_ids, args.temperature)
        with torch.no_grad(), model.disable_adapter():
            ref = token_logprobs(model, prompt_ids, completion_ids, args.temperature)
        ratio = torch.exp(logp - torch.tensor(old, device=logp.device))
        surrogate = torch.min(ratio * advantage, ratio.clamp(1 - args.clip_low, 1 + args.clip_high) * advantage)
        kl = (torch.exp(ref - logp) - (ref - logp) - 1).clamp(max=10)  # verl's low_var_kl
        ((args.kl * kl - surrogate).sum() / n_tokens).backward()
        stats["tokens"] += len(completion_ids)
        stats["kl"] += kl.sum().item()
        stats["clipped"] += ((ratio - 1).abs() > args.clip_low).sum().item()
        stats["logp_gap"] += (ratio.log()).abs().sum().item()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/nq_hotpotqa_train/train.parquet")
    parser.add_argument("--search", choices=["local", "wikipedia"], default="local",
                        help="Wikipedia_Search_Tool: Search-R1's local retriever or the official tool on live Wikipedia")
    parser.add_argument("--retrieval", default="data/retrieval")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--questions", type=int, default=32, help="questions per step (AgentFlow: 32)")
    parser.add_argument("--group", type=int, default=8, help="rollouts per question (AgentFlow: 8)")
    parser.add_argument("--minibatches", type=int, default=4, help="updates per step (AgentFlow: 32 / 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate (AgentFlow, full: 1e-6)")
    parser.add_argument("--kl", type=float, default=0.001)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=3, help="AgentFlow turns per rollout in training")
    parser.add_argument("--temperature", type=float, default=0.7, help="planner sampling temperature")
    parser.add_argument("--module-temperature", type=float,
                        help="temperature of the frozen modules (default: the planner's, as the official code); 0 "
                             "makes the reward differences within a group come from the planner alone")
    parser.add_argument("--keep-every", type=int, default=10, help="steps between adapters kept for evaluation")
    parser.add_argument("--vllm-mem", type=float, default=0.4)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpoint = last_checkpoint(out) if args.resume else None
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear"))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    start = 0
    if checkpoint:
        state = torch.load(checkpoint / "state.pt")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        print(f"resuming from {checkpoint} (step {start})", flush=True)
    wikipedia = search_tool(args)  # before vLLM, which then takes its share of the GPU
    llm = LLM(args.model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=args.vllm_mem,
              seed=args.seed, enable_lora=True, max_lora_rank=16, logprobs_mode="processed_logprobs")
    questions = load_questions(args.data, args.seed)

    for step in range(start, args.steps):
        t0 = time.time()
        live = out / "live_adapter"
        model.save_pretrained(live)
        batch = [questions[(step * args.questions + i) % len(questions)] for i in range(args.questions)]
        prompts = [f"{q.strip()} {agentflow.ANSWER_FORMAT}" for q, _ in batch for _ in range(args.group)]
        episodes = agentflow.solve(llm, tokenizer, wikipedia, prompts, args.max_steps, args.temperature,
                                   LoRARequest(f"step_{step}", step + 1, str(live)),
                                   module_temperature=args.module_temperature)
        t1 = time.time()
        rewards = [agentflow.cover_match(agentflow.final_answer(ep.output), g)
                   for ep, (_, g) in zip(episodes, [b for b in batch for _ in range(args.group)])]
        r = torch.tensor(rewards).view(args.questions, args.group)
        advantages = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).flatten().tolist()

        model.train()
        order = list(range(len(episodes)))
        random.shuffle(order)
        stats = Counter()
        for mb in range(args.minibatches):
            chosen = [episodes[i] for i in order[mb::args.minibatches]]
            adv = [advantages[i] for i in order[mb::args.minibatches]]
            # Token-mean over every planner token of the minibatch. Turns of rollouts with advantage 0
            # (a group with all-equal rewards) would only add the tiny KL term, so they are not run.
            n_tokens = sum(len(t[1]) for ep in chosen for t in ep.turns)
            samples = [(*turn, a) for ep, a in zip(chosen, adv) if a != 0 for turn in ep.turns if turn[1]]
            stats.update(policy_loss(model, samples, max(n_tokens, 1), args))
            if mb == 0:  # before any update this step: only the vLLM / training numerical gap
                first_gap = stats["logp_gap"] / stats["tokens"] if stats["tokens"] else None
            stats["grad_norm"] += torch.nn.utils.clip_grad_norm_(params, 1.0).item() / args.minibatches
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        save_checkpoint(model, optimizer, out, step + 1)
        if (step + 1) % args.keep_every == 0 or step + 1 == args.steps:
            model.save_pretrained(out / "adapters" / f"step_{step + 1:04d}")

        turns = [len(ep.turns) for ep in episodes]
        tools = Counter(m["tool_name"] for ep in episodes for m in ep.memory.values())
        planner_tokens = [len(t[1]) for ep in episodes for t in ep.turns]
        trained = max(stats["tokens"], 1)
        metrics = {"step": step + 1, "reward": r.mean().item(),
                   "frac_zero_std": (r.std(1) == 0).float().mean().item(),
                   "turns": sum(turns) / len(turns), "planner_tokens": sum(planner_tokens) / len(planner_tokens),
                   "tools": dict(tools), "kl": stats["kl"] / trained, "clip_frac": stats["clipped"] / trained,
                   "logp_gap": stats["logp_gap"] / trained, "logp_gap_first": first_gap, "grad_norm": stats["grad_norm"],
                   "rollout_s": t1 - t0, "train_s": time.time() - t1}
        print(json.dumps(metrics), flush=True)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    main()

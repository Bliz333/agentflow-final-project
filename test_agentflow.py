"""Checks that need no GPU model: the AgentFlow loop with a scripted LLM, and the Flow-GRPO update on a
tiny random model.    python -m pytest -q test_agentflow.py
"""
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config

import agentflow
import train

PLAN = """1.  **Justification:** I need a fact.
2.  **Context:** The query asks about {topic}.
3.  **Sub-Goal:** Find {topic}.
4.  **Tool Name:** {tool}"""


class ScriptedLLM:
    """Answers each module's prompt like the official format; question "B" never satisfies the verifier."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        outs = []
        for p in prompts:
            if "Determine the optimal next step" in p:
                tool = "Wikipedia Search Tool" if "Action Step 1" not in p else "base_generator_tool"
                text = PLAN.format(topic="the founding year", tool=tool)
                module = "planner"
            elif "Generate a precise command" in p:
                text = 'Generated Command:\n```python\nexecution = tool.execute(query="founding year")\n```'
                module = "executor"
            elif "Evaluate if the current memory" in p:
                text = "Missing info.\nConclusion: CONTINUE" if "Question B" in p else "Enough.\nConclusion: STOP"
                module = "verifier"
            elif "Generate a concise final answer" in p:
                text = "**Answer:** <answer>1812</answer>"
                module = "generator"
            elif "Analyze the given query" in p:
                text, module = "Needs one search.", "analysis"
            else:
                text, module = "The base model's answer.", "base_generator"
            self.calls.append((module, lora_request is not None, params.temperature))
            ids = [1, 2, 3]
            logprobs = [{t: SimpleNamespace(logprob=-0.5)} for t in ids] if params.logprobs is not None else None
            outs.append(SimpleNamespace(prompt_token_ids=[7, 8], outputs=[SimpleNamespace(
                text=text, token_ids=ids, logprobs=logprobs)]))
        return outs


class FakeRetriever:
    def search(self, queries, topk=3):
        return [[f'"Page {q}"\ntext about {q}'] for q in queries]


def test_parse_plan_matches_tool_names_loosely():
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="Wikipedia_Search_Tool")) == \
        ("The query asks about x.", "Find x.", "Wikipedia_Search_Tool")
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="base generator tool"))[2] == "Base_Generator_Tool"
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="Google_Search_Tool"))[2] is None
    assert agentflow.parse_plan("no structure at all") == ("", "", None)


@pytest.mark.parametrize("command,query", [
    ('execution = tool.execute(query="Who founded Citibank?")', "Who founded Citibank?"),
    ("execution = tool.execute(query='Citibank')", "Citibank"),
    ('execution = tool.execute(query="""multi\nline""")', "multi\nline"),
])
def test_command_query(command, query):
    assert agentflow.COMMAND.search(command).group(2) == query


def test_solve_runs_modules_in_order_and_stops():
    llm = ScriptedLLM()
    tokenizer = SimpleNamespace(apply_chat_template=lambda m, **kw: "\n".join(x["content"] for x in m))
    lora = object()
    episodes = agentflow.solve(llm, tokenizer, FakeRetriever(), ["Question A", "Question B"], max_steps=3,
                               temperature=0.7, lora_request=lora)
    a, b = episodes
    assert len(a.memory) == 1 and len(b.memory) == 3          # A stops after the verifier's STOP
    assert a.memory["Action Step 1"]["tool_name"] == "Wikipedia_Search_Tool"
    assert "text about founding year" in a.memory["Action Step 1"]["result"]
    assert b.memory["Action Step 2"]["result"] == "The base model's answer."
    assert len(a.turns) == 1 and len(b.turns) == 3            # one trained planner sample per turn
    assert b.turns[0] == ([7, 8], [1, 2, 3], [-0.5, -0.5, -0.5])
    assert agentflow.final_answer(a.output) == "1812"
    # Only the planner uses the LoRA; every module samples, the generator tool is greedy; no verifier
    # call after the last turn.
    assert {(m, lora_used, t) for m, lora_used, t in llm.calls} == {
        ("planner", True, 0.7), ("analysis", False, 0.7), ("executor", False, 0.7), ("verifier", False, 0.7),
        ("generator", False, 0.7), ("base_generator", False, 0.0)}
    assert sum(m == "verifier" for m, _, _ in llm.calls) == 2 + 1  # step 1: A and B, step 2: B


def test_unknown_tool_and_bad_command_become_error_results():
    results = agentflow.run_tools(None, None, FakeRetriever(), [(None, "x"), ("Wikipedia_Search_Tool", "search it")])
    assert results[0].startswith("Error:") and results[1].startswith("Error:")


def test_judge_reads_the_verdict_and_asks_once_per_answer(monkeypatch):
    prompts = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            verdict = {"yes": "<analysis>: same\n<true_false>: True", "no": '<true_false>: "False"'}
            reply = next((v for k, v in verdict.items() if f"Model Response: {k}" in prompts[-1]), "?")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test")
    monkeypatch.setattr("openai.OpenAI", Client)
    assert agentflow.judge(["q", "q", "q", "p", "q"], ["yes", "no", "garbled", "yes", "yes"], [["a"]] * 5) == \
        [1.0, 0.0, 0.0, 1.0, 1.0]
    assert len(prompts) == 4                                    # ("q", "yes") is judged once
    assert agentflow.judge(["q"], ["The A."], [["a"]]) == [1.0] and len(prompts) == 4  # exact match: no call


def test_exact_match_and_cover_match():
    assert agentflow.exact_match("The James Madison.", ["james madison"]) == 1.0
    assert agentflow.exact_match("Madison", ["james madison"]) == 0.0
    assert agentflow.cover_match("Richard Nixon died on April 22, 1994.", ["April 22, 1994"]) == 1.0
    assert agentflow.cover_match("8 January 1891", ["January 8, 1891"]) == 0.0
    assert agentflow.cover_match("anything", ["", "the"]) == 0.0           # empty after normalisation


def tiny_policy():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(config)
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"))


def test_policy_loss_moves_logprob_with_the_advantage():
    model = tiny_policy()
    args = SimpleNamespace(temperature=0.7, clip_low=0.2, clip_high=0.3, kl=0.001)
    prompt, completion = [1, 2, 3, 4], [5, 6, 7]
    with torch.no_grad():
        old = train.token_logprobs(model, prompt, completion, args.temperature).tolist()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    stats = train.policy_loss(model, [(prompt, completion, old, 1.0)], 3, args)
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6)        # a fresh LoRA is the base model
    assert stats["clipped"] == 0 and stats["logp_gap"] == pytest.approx(0.0, abs=1e-5)
    optimizer.step()
    with torch.no_grad():
        new = train.token_logprobs(model, prompt, completion, args.temperature)
    assert new.sum() > sum(old)                                 # positive advantage: more likely
    optimizer.zero_grad()
    train.policy_loss(model, [(prompt, completion, new.tolist(), -1.0)], 3, args)
    optimizer.step()
    with torch.no_grad():
        assert train.token_logprobs(model, prompt, completion, args.temperature).sum() < new.sum()


def test_checkpoints_keep_only_the_last(tmp_path):
    model = tiny_policy()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    train.save_checkpoint(model, optimizer, tmp_path, 1)
    train.save_checkpoint(model, optimizer, tmp_path, 2)
    (tmp_path / "checkpoint_0003").mkdir()                      # killed before state.pt was written
    assert train.last_checkpoint(tmp_path) == tmp_path / "checkpoint_0002"
    assert not (tmp_path / "checkpoint_0001").exists()
    assert torch.load(tmp_path / "checkpoint_0002" / "state.pt")["step"] == 2

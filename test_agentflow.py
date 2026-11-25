"""Checks that need no GPU model: the AgentFlow loop with a scripted LLM, and the Flow-GRPO update on a
tiny random model.    python -m pytest -q test_agentflow.py
"""
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config

import agentflow
import google_search
import train
import wikipedia

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
    tools = ["Base_Generator_Tool", "Wikipedia_Search_Tool"]
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="Wikipedia_Search_Tool"), tools) == \
        ("The query asks about x.", "Find x.", "Wikipedia_Search_Tool")
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="base generator tool"), tools)[2] == "Base_Generator_Tool"
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="Google_Search_Tool"), tools)[2] is None
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="Google Search"), tools + ["Google_Search_Tool"])[2] == \
        "Google_Search_Tool"
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="wikipedia_search"), tools)[2] == "Wikipedia_Search_Tool"
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="Tool"), tools)[2] is None
    assert agentflow.parse_plan(PLAN.format(topic="x", tool="Wikipedia"), tools)[2] == "Wikipedia_Search_Tool"
    assert agentflow.parse_plan("no structure at all", tools) == ("", "", None)


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
    episodes = agentflow.solve(llm, tokenizer, agentflow.LocalWikipedia(FakeRetriever()), ["Question A", "Question B"], max_steps=3,
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
    llm.calls.clear()
    agentflow.solve(llm, tokenizer, agentflow.LocalWikipedia(FakeRetriever()), ["Question A"], 3, 0.7, lora, module_temperature=0.0)
    assert {(m, t) for m, _, t in llm.calls} == {("planner", 0.7), ("analysis", 0.0), ("executor", 0.0),
                                                 ("verifier", 0.0), ("generator", 0.0)}


def test_verifier_conclusion_ignores_case():
    assert [v.upper() for v in agentflow.CONCLUSION.findall("Enough.\nConclusion: Stop")] == ["STOP"]


def test_unknown_tool_and_bad_command_become_error_results():
    results = agentflow.run_tools(None, None, agentflow.LocalWikipedia(FakeRetriever()), [(None, "x"), ("Wikipedia_Search_Tool", "search it")])
    assert results[0].startswith("Error:") and results[1].startswith("Error:")


class FakeGoogle:
    metadata = google_search.METADATA

    def __init__(self):
        self.queries = []

    def run(self, queries):
        self.queries += queries
        return [f"web answer to {q}" for q in queries]


def test_google_search_tool_is_offered_and_called_only_when_given():
    wiki, google = agentflow.LocalWikipedia(FakeRetriever()), FakeGoogle()
    calls = [("Google_Search_Tool", 'execution = tool.execute(query="founding year")'),
             ("Wikipedia_Search_Tool", 'execution = tool.execute(query="Citibank")')]
    assert agentflow.run_tools(None, None, wiki, calls)[0].startswith("Error:")
    results = agentflow.run_tools(None, None, wiki, calls, google)
    assert results[0] == "web answer to founding year" and "text about Citibank" in results[1]
    assert google.queries == ["founding year"]
    llm, prompts = ScriptedLLM(), []
    generate = llm.generate
    llm.generate = lambda ps, *args, **kwargs: (prompts.extend(ps), generate(ps, *args, **kwargs))[1]
    tokenizer = SimpleNamespace(apply_chat_template=lambda m, **kw: "\n".join(x["content"] for x in m))
    agentflow.solve(llm, tokenizer, wiki, ["Question A"], 1, 0.7, google=google)
    assert "['Base_Generator_Tool', 'Google_Search_Tool', 'Wikipedia_Search_Tool']" in prompts[0]
    assert "Gemini" in prompts[0]
    prompts.clear()
    agentflow.solve(llm, tokenizer, wiki, ["Question A"], 1, 0.7)
    assert "['Base_Generator_Tool', 'Wikipedia_Search_Tool']" in prompts[0] and "Gemini" not in prompts[0]


def test_google_search_asks_deepseek_with_web_search_and_cites_like_the_official_tool(monkeypatch):
    requests = []
    cited = lambda url, end: SimpleNamespace(type="url_citation", url_citation=SimpleNamespace(url=url, end_index=end))
    text = "Paris is the capital. It is in France."  # the two sentences end at 21 and 38

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=text, annotations=[cited("u1", 21), cited("u2", 21), cited("u1", 38)]))])

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(google_search, "OpenAI", Client)
    assert google_search.DeepSeekSearch().run(["capital of France?"]) == \
        ["Paris is the capital.[1](u1), [2](u2) It is in France.[1](u1)"]
    assert requests == [{"model": "deepseek-flash", "messages": [{"role": "user", "content": "capital of France?"}],
                         "tools": [{"type": "deepseek_web_search"}],
                         "extra_body": {"thinking": {"type": "disabled"}}}]
    assert google_search.cite("no sources", None) == "no sources"


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
    assert agentflow.cover_match("January 18, 1988", ["8"]) == 0.0         # whole words only
    assert agentflow.cover_match("It is 8.", ["8", "eight"]) == 1.0


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
    assert not (tmp_path / "checkpoint_0002" / "state.tmp").exists()


def test_chunks_slide_200_words_with_20_overlap():
    words = [f"w{i}" for i in range(400)]
    out = wikipedia.chunks(" ".join(words))
    assert [c.split()[0] for c in out] == ["w0", "w180", "w360"] and len(out[0].split()) == 200
    assert wikipedia.chunks("") == [] and wikipedia.chunks("a b") == ["a b"]


def test_live_wikipedia_picks_pages_ranks_chunks_and_summarises(monkeypatch):
    hits = {"q1": (("Citigroup", "bank"), ("Moon", "satellite"), ("James Madison", "president")), "q2": ()}
    pages = {"Citigroup": "Citibank was founded in 1812 " * 50, "James Madison": "fourth president " * 300,
             "Moon": ""}
    monkeypatch.setattr(wikipedia, "search", lambda q: hits[q])
    monkeypatch.setattr(wikipedia, "page_text", lambda title: pages.get(title, ""))
    tool = wikipedia.LiveWikipedia.__new__(wikipedia.LiveWikipedia)
    tool.workers, tool.top_k = 2, 10
    tool.embed = lambda texts, prefix: torch.nn.functional.normalize(torch.rand(len(texts), 4), dim=-1)
    prompts = []

    class LLM:
        def generate(self, ps, params, lora_request=None, use_tqdm=False):
            prompts.extend(ps)
            reply = lambda p: ("- Matched Queries: Citigroup, Moon, James Madison\n- Matched Query IDs: [0, 1, 2, 7]"
                               if "Query Candidates" in p else "Answer:\nsummary")
            return [SimpleNamespace(outputs=[SimpleNamespace(text=reply(p))]) for p in ps]

    tokenizer = SimpleNamespace(apply_chat_template=lambda m, **kw: m[-1]["content"])
    r1, r2 = tool.run(["q1", "q2"], LLM(), tokenizer)
    relevant = r1["relevant_pages (to the query)"]
    assert [p["title"] for p in relevant] == ["Citigroup", "Moon", "James Madison"]  # id 7 does not exist
    assert [p["retrieved_information"] for p in relevant] == ["Answer:\nsummary", None, "Answer:\nsummary"]
    assert r1["other_pages (may be irrelevant to the query)"] == []
    assert sum("Query Candidates" in p for p in prompts) == 1                # nothing to pick from for q2
    assert r2 == {"query": "q2", "relevant_pages (to the query)": [], "other_pages (may be irrelevant to the query)": []}
    summaries = [p for p in prompts if "### User Query" in p]
    assert [p.count("Chunk [") for p in summaries] == [2, 4]                # all chunks when fewer than 10


def test_wikipedia_api_waits_on_429_and_raises_on_other_errors(monkeypatch):
    import io
    import urllib.error
    calls, slept = [], []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "2"}, None)
        if "fail" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, None)
        return io.BytesIO(b'{"query": {"search": [{"title": "Moon", "snippet": "<b>Moon</b> rocks"}]}}')

    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(wikipedia.time, "sleep", slept.append)
    assert wikipedia.api(srsearch="moon")["query"]["search"][0]["title"] == "Moon"
    assert len(calls) == 2 and slept == [2.0]
    with pytest.raises(urllib.error.HTTPError):
        wikipedia.api(srsearch="fail")

"""AgentFlow (Li et al., 2025, arXiv 2510.05592) for short-answer questions, every module run by one vLLM engine.

Per question the planner first analyses the query. Then, each turn, the planner picks a sub-goal and a
tool, the executor writes the tool call, the tool runs, the call and its result go into the memory, and
the verifier says whether the memory answers the query (STOP) or not (CONTINUE). After STOP or the last
turn the solution generator writes the answer from the memory. The prompts are the official ones
(github.com/lupantech/AgentFlow, agentflow/models/{planner,executor,verifier}.py, text-only branch).

As in the official code, only the planner's next-step calls are the trained policy (they get the LoRA
through `lora_request`); the query analysis, executor, verifier and generator are the frozen
Qwen2.5-7B-Instruct, and every module samples at the same temperature (the official solver passes one
temperature to all of them); the Base_Generator tool runs at temperature 0, as the paper's tool engines
do. All questions advance together: each module call is one batched vLLM generate over the questions
that are still running.

Tools: Base_Generator_Tool (the base model answers the tool query), Wikipedia_Search_Tool, either the
official tool on live Wikipedia (wikipedia.LiveWikipedia) or Search-R1's E5 retriever over the 2018
Wikipedia dump (LocalWikipedia, top 3 passages), and optionally Google_Search_Tool, the official tool with
DeepSeek's web search in place of Gemini 2.5 Flash with Google grounding (google_search.DeepSeekSearch).
"""
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from retriever import format_passages

SYSTEM = "You are a helpful, creative, and smart assistant."  # official vLLM engine's system prompt
ANSWER_FORMAT = ("When ready, output the final answer enclosed in <answer> and </answer> tags. "
                 "Do not generate any content after the </answer> tag.")  # appended to every question

TOOLBOX = {
    "Base_Generator_Tool": {
        "tool_description": "A generalized tool that takes query from the user, and answers the question step by "
                            "step to the best of its ability.",
        "input_types": {"query": "str - The query that includes query from the user to guide the agent to "
                                 "generate response."},
        "output_type": "str - The generated response to the original query",
        "demo_commands": [{"command": 'execution = tool.execute(query="Summarize the following text in a few lines")',
                           "description": "Generate a short summary given the query from the user."}],
    },
}  # and the search backends' metadata, see solve()


class LocalWikipedia:
    """Wikipedia_Search_Tool on Search-R1's E5 retriever over the 2018 Wikipedia dump: the top 3 passages."""
    metadata = {
        "tool_description": "A tool that searches Wikipedia and returns the 3 most relevant passages, with their "
                            "page titles, for a given query.",
        "input_types": {"query": "str - The search query for Wikipedia."},
        "output_type": "str - The 3 most relevant Wikipedia passages, each with its page title.",
        "demo_commands": [
            {"command": 'execution = tool.execute(query="What is the exact mass in kg of the moon")',
             "description": "Search Wikipedia and get the information about the mass of the moon."},
            {"command": 'execution = tool.execute(query="When was the first moon landing?")',
             "description": "Search Wikipedia and get the information about the first moon landing."},
        ],
    }

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, queries, llm, tokenizer):
        return [format_passages(p) for p in self.retriever.search(queries)]

QUERY_ANALYSIS = """
Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {tools}
- Metadata for tools: {toolbox}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

Be biref and precise with insight.
"""

NEXT_STEP = """
Task: Determine the optimal next step to address the query using available tools and previous steps.

Context:
- **Query:** {question}
- **Query Analysis:** {analysis}
- **Available Tools:** {tools}
- **Toolbox Metadata:** {toolbox}
- **Previous Steps:** {memory}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1.  **Justification:** Explain your choice of tool and sub-goal.
2.  **Context:** Provide all necessary information for the tool.
3.  **Sub-Goal:** State the specific objective for the tool.
4.  **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content.
"""

TOOL_COMMAND = """
Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {question}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example1:
Generated Command:
```python
execution = tool.execute(query="Summarize the following porblom:"Isaac has 100 toys, masa gets ...., how much are their together?")
```

Example2:
Generated Command:
```python
execution = tool.execute(query=["Methanol", "function of hyperbola", "Fermat's Last Theorem"])
```
"""

VERIFY = """
Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {tools}
- **Toolbox Metadata:** {toolbox}
- **Initial Analysis:** {analysis}
- **Memory (Tools Used & Results):** {memory}

Instructions:
1.  Review the query, initial analysis, and memory.
2.  Assess the completeness of the memory: Does it fully address all parts of the query?
3.  Check for potential issues:
    -   Are there any inconsistencies or contradictions?
    -   Is any information ambiguous or in need of verification?
4.  Determine if any unused tools could provide missing information.

Final Determination:
-   If the memory is sufficient, explain why and conclude with "STOP".
-   If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""

# Solution generator: the official "direct" output (generate_direct_output), used for all QA benchmarks.
SOLUTION = """
Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {analysis}
- **Actions Taken:** {memory}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

Output Structure:
1.  **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2.  **Answer:** A direct and concise final answer to the query.
"""

# The official judge prompt (train/utils.py), run by DeepSeek V4.1 Flash instead of GPT-4o.
JUDGE_URL = "https://opencode.ai/zen/go/v1"  # OpenCode Go, OpenAI-compatible; key in OPENCODE_GO_API_KEY
JUDGE_MODEL = "deepseek-v4.1-flash"
JUDGE = """
You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

**Instructions:**
1.  **Extract:** Isolate the final answer from the Model Response, ignoring reasoning. Look for `\\boxed{{...}}` or concluding statements.
2.  **Normalize & Compare:** The extracted answer and Ground Truth must be equivalent after normalization:
    - **Math:** Mathematically identical (e.g., `\\\\frac{{1}}{{2}}` == `0.5`).
    - **Numbers/Text:** Ignore formatting, case, and currency/units (e.g., `1,000` == `1000`).
    - **MCQ:** Match option content (e.g., "Paris") or number (e.g., `3rd` option) to the correct letter.
3.  **Verdict:** "True" only for semantically or mathematically equivalent answers.

**Inputs:**
Question: {question}
Model Response: {response}
Ground Truth: {truth}

**Format:**
<analysis>: Brief analysis of the comparison.
<true_false>: "True" or "False".
"""

# The official fallback parser (planner.py), also skipping the "3." / "4." numbering of the response format.
PLAN = re.compile(r"Context:\s*(.*?)\s*(?:\d\.\s*)?Sub-Goal:\s*(.*?)\s*(?:\d\.\s*)?Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)",
                  re.DOTALL)
COMMAND = re.compile(r"tool\.execute\(\s*query\s*=\s*(\"\"\"|'''|\"|')(.*?)\1\s*\)", re.DOTALL)
CONCLUSION = re.compile(r"Conclusion\W*(STOP|CONTINUE)", re.IGNORECASE)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
VERDICT = re.compile(r"true_false\W*(True|False)", re.IGNORECASE)


@dataclass
class Episode:
    question: str
    analysis: str = ""
    memory: dict = field(default_factory=dict)  # "Action Step n" -> tool_name, sub_goal, command, result
    turns: list = field(default_factory=list)   # planner next-step calls: (prompt_ids, completion_ids, logprobs)
    output: str = ""                            # solution generator output
    done: bool = False


def chat(tokenizer, text):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(llm, prompts, max_tokens, temperature=0.0, lora_request=None):
    from vllm import SamplingParams
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=0 if lora_request else None)
    return llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)


def parse_plan(text, tools):
    """(context, sub_goal, tool_name) from the planner's answer, as the official regex fallback does; tool_name
    is one of `tools` or None."""
    matches = PLAN.findall(text.replace("**", ""))
    if not matches:
        return "", "", None
    context, sub_goal, tool = (s.strip() for s in matches[-1])
    key = tool.lower().replace(" ", "_")  # e.g. "Wikipedia Search Tool", "wikipedia_search", "Wikipedia"
    return context, sub_goal, next((t for t in tools if t.lower().removesuffix("_tool") in key
                                    or key and t.lower().startswith(key)), None)


RESULT_CHARS = 8000  # a longer tool result is cut, as the official code cuts at 100,000 for its 128k-token engines:
                     # a few web search answers reach 36,000 characters and overflowed the 32k-token context


def run_tools(llm, tokenizer, wikipedia, calls, google=None):
    """Results of (tool_name, command) calls; each tool's calls run as one batch."""
    tools = {"Base_Generator_Tool": lambda qs: [o.outputs[0].text for o in
                                                generate(llm, [chat(tokenizer, q) for q in qs], 1024)],
             "Wikipedia_Search_Tool": lambda qs: wikipedia.run(qs, llm, tokenizer)}
    if google:
        tools["Google_Search_Tool"] = google.run
    queries = [COMMAND.search(command) for _, command in calls]
    results = [f"Error: {tool!r} is not an available tool." if tool not in tools else
               "Error: the command has no execution = tool.execute(query=\"...\") call." for tool, _ in calls]
    for name, run in tools.items():
        idx = [i for i, ((tool, _), q) in enumerate(zip(calls, queries)) if tool == name and q]
        if not idx:
            continue
        for i, out in zip(idx, run([queries[i].group(2) for i in idx])):
            results[i] = out[:RESULT_CHARS] + "..." if isinstance(out, str) and len(out) > RESULT_CHARS else out
    return results


def solve(llm, tokenizer, wikipedia, questions, max_steps, temperature, lora_request=None, planner_tokens=2048,
          module_temperature=None, google=None, planner_llm=None):
    """Run AgentFlow on questions (strings, ANSWER_FORMAT already appended); one Episode per question.

    `temperature` is the planner's; the other modules use `module_temperature` (default: the same).
    `google` is the Google_Search_Tool's backend (none = the tool is left out). `planner_llm` runs the planner's
    next-step calls (default: `llm`), e.g. the paper's released planner, a full model rather than a LoRA."""
    module_temperature = temperature if module_temperature is None else module_temperature
    toolbox = {**TOOLBOX, **({"Google_Search_Tool": google.metadata} if google else {}),
               "Wikipedia_Search_Tool": wikipedia.metadata}  # the official order
    fill = dict(tools=list(toolbox), toolbox=toolbox)
    episodes = [Episode(q) for q in questions]
    for ep, out in zip(episodes, generate(llm, [chat(tokenizer, QUERY_ANALYSIS.format(question=q, **fill))
                                               for q in questions], 1024, module_temperature)):
        ep.analysis = out.outputs[0].text
    for step in range(1, max_steps + 1):
        active = [ep for ep in episodes if not ep.done]
        if not active:
            break
        prompts = [chat(tokenizer, NEXT_STEP.format(question=ep.question, analysis=ep.analysis, memory=ep.memory,
                                                    **fill)) for ep in active]
        plans = []
        for ep, out in zip(active, generate(planner_llm or llm, prompts, planner_tokens, temperature, lora_request)):
            gen = out.outputs[0]
            ep.turns.append((list(out.prompt_token_ids), list(gen.token_ids),
                             [lp[t].logprob for lp, t in zip(gen.logprobs, gen.token_ids)] if gen.logprobs else None))
            plans.append(parse_plan(gen.text, fill["tools"]))
        prompts = [chat(tokenizer, TOOL_COMMAND.format(question=ep.question, sub_goal=sub_goal, tool_name=tool,
                                                       tool_metadata=toolbox.get(tool), context=context))
                   for ep, (context, sub_goal, tool) in zip(active, plans)]
        commands = [o.outputs[0].text.split("Generated Command:")[-1].strip()
                    for o in generate(llm, prompts, 512, module_temperature)]
        results = run_tools(llm, tokenizer, wikipedia, [(tool, c) for (_, _, tool), c in zip(plans, commands)], google)
        for ep, (_, sub_goal, tool), command, result in zip(active, plans, commands, results):
            ep.memory[f"Action Step {step}"] = {"tool_name": tool, "sub_goal": sub_goal, "command": command,
                                                "result": result}
        if step == max_steps:  # the verifier's verdict after the last turn would not change anything
            break
        prompts = [chat(tokenizer, VERIFY.format(question=ep.question, analysis=ep.analysis, memory=ep.memory,
                                                 **fill)) for ep in active]
        for ep, out in zip(active, generate(llm, prompts, 1024, module_temperature)):
            verdicts = CONCLUSION.findall(out.outputs[0].text)
            ep.done = bool(verdicts) and verdicts[-1].upper() == "STOP"
    prompts = [chat(tokenizer, SOLUTION.format(question=ep.question, analysis=ep.analysis, memory=ep.memory))
               for ep in episodes]
    for ep, out in zip(episodes, generate(llm, prompts, 1024, module_temperature)):
        ep.output = out.outputs[0].text
    return episodes


def final_answer(output):
    """The last <answer> ... </answer>, else the whole output (what the official code sends to the judge)."""
    matches = ANSWER.findall(output)
    return matches[-1].strip() if matches else output.strip()


def judge(questions, responses, golden_answers, workers=8):
    """1.0 / 0.0 per response: the official judge prompt, DeepSeek V4.1 Flash, greedy, without thinking.

    The API answers about 40 requests a minute (a refused request, 429, is retried with backoff), so an
    exact match is correct without asking, and a response repeated for the same question is asked once."""
    from openai import OpenAI
    client = OpenAI(base_url=JUDGE_URL, api_key=os.environ["OPENCODE_GO_API_KEY"], timeout=120, max_retries=100,
                    default_headers={"x-opencode-session": f"info7375-w9-{os.getpid()}"})

    def one(item):
        question, response, truth = item
        if exact_match(response, truth):
            return 1.0
        reply = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0, extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": JUDGE.format(question=question, response=response, truth=truth)}])
        verdict = VERDICT.findall(reply.choices[0].message.content or "")
        return float(bool(verdict) and verdict[-1].lower() == "true")

    items = list(zip(questions, responses, [list(g) for g in golden_answers]))
    unique = list({(q, r): (q, r, g) for q, r, g in items}.values())
    with ThreadPoolExecutor(workers) as pool:
        verdicts = {(q, r): v for (q, r, _), v in zip(unique, pool.map(one, unique))}
    return [verdicts[q, r] for q, r, _ in items]


def normalize_answer(s):
    """SQuAD / FlashRAG normalisation: lower case, no punctuation, no articles, single spaces."""
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def exact_match(prediction, golden_answers):
    return float(any(normalize_answer(g) == normalize_answer(prediction) for g in golden_answers))


def cover_match(prediction, golden_answers):
    """FlashRAG's accuracy, on whole words: a golden answer appears in the normalised prediction (the
    training reward). Whole words, so that a golden "8" is not found in "1988"."""
    prediction = f" {normalize_answer(prediction)} "
    return float(any(normalize_answer(g) and f" {normalize_answer(g)} " in prediction for g in golden_answers))

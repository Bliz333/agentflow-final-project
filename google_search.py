"""The official AgentFlow Google_Search_Tool (agentflow/tools/google_search/tool.py) with DeepSeek's web search
in place of Gemini's Google grounding.

The official tool sends the query, as it is, to gemini-2.5-flash with Google Search grounding and returns
the answer with "[n](url)" after each cited span. Here the query goes, as it is, to deepseek-flash
(DeepSeek-V4.1-Flash) with DeepSeek's server-side web search (`deepseek_web_search`, thinking off: the
model searches as needed and answers with citations), and the citations are inserted the same way. The
planner sees the official metadata, without the `add_citations` argument (always on, its default).
Gemini 2.5 Flash is closed to new API users and is retired on 1 December 2026.
"""
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# Official metadata (google_search/tool.py).
METADATA = {
    "tool_description": "A web search tool powered by Google's Gemini AI that provides real-time information from "
                        "the internet with citation support.",
    "input_types": {"query": "str - The search query to find information on the web."},
    "output_type": "str - The search results of the query.",
    "demo_commands": [
        {"command": 'execution = tool.execute(query="What is the capital of France?")',
         "description": "Search for general information about the capital of France with default citations enabled."},
        {"command": 'execution = tool.execute(query="Physics and Society article arXiv August 11, 2016")',
         "description": "Search for specific academic articles with citations enabled."},
    ],
}


def cite(text, annotations):
    """The official add_citations: after each cited span, "[n](url)" for each of its sources, n numbering the
    distinct URLs in order of appearance."""
    spans = [a.url_citation for a in annotations or [] if a.type == "url_citation"]
    number, ends = {}, {}
    for s in spans:
        n = number.setdefault(s.url, len(number) + 1)
        ends.setdefault(s.end_index, []).append(f"[{n}]({s.url})")
    for end in sorted(ends, reverse=True):
        text = text[:end] + ", ".join(ends[end]) + text[end:]
    return text


class DeepSeekSearch:
    metadata = METADATA

    def __init__(self, workers=32, model="deepseek-flash"):
        self.client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
        self.workers, self.model = workers, model

    def search(self, query):
        reply = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": query}], tools=[{"type": "deepseek_web_search"}],
            extra_body={"thinking": {"type": "disabled"}})
        message = reply.choices[0].message
        return cite(message.content or "", message.annotations)

    def run(self, queries):
        """One search per distinct query of the batch: in training, the 8 rollouts of a question often send the
        same query in the same turn, and each search costs about 5k tokens of search results."""
        distinct = list(dict.fromkeys(queries))
        with ThreadPoolExecutor(self.workers) as pool:
            answers = dict(zip(distinct, pool.map(self.search, distinct)))
        return [answers[q] for q in queries]

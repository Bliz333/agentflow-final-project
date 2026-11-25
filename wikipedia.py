"""The official AgentFlow Wikipedia_Search_Tool (agentflow/tools/wikipedia_search/tool.py, which hands every
picked page to agentflow/tools/web_search/tool.py), run for many queries at once.

Per query: Wikipedia's search gives 10 titles; the LLM picks up to 3 of them (official prompt); each picked
page's text is cut into 200-word chunks overlapping by 20 words; the 10 chunks most similar to the query
are kept; the LLM answers the query from them (official prompt). The result is the official dict: the
picked pages with that answer, the other pages with a short abstract.

Differences from the official code: page text comes from the MediaWiki API as plain text instead of
scraping the page's HTML, and abstracts are the search snippets; chunks are ranked with E5-base-v2 instead
of OpenAI's text-embedding-3-large; the LLM is the base model, the paper's tool engine, instead of
gpt-4o-mini. In the released official code the per-page step never runs (the tool's model_string is never
set, so creating the page tool raises and the tool returns only the titles and abstracts); this is the
behaviour that code intends.
"""
import html
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch
import torch.nn.functional as F

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "agentflow-final-project (https://github.com/Bliz333/agentflow-final-project)"
ENCODER = "intfloat/e5-base-v2"

METADATA = {
    "tool_description": "A tool that searches Wikipedia and returns relevant pages with their page titles, URLs, "
                        "abstract, and retrieved information based on a given query.",
    "input_types": {"query": "str - The search query for Wikipedia."},
    "output_type": "dict - A dictionary containing search results, all matching pages with their content, URLs, "
                   "and metadata.",
    "demo_commands": [
        {"command": 'execution = tool.execute(query="What is the exact mass in kg of the moon")',
         "description": "Search Wikipedia and get the information about the mass of the moon."},
        {"command": 'execution = tool.execute(query="When was the first moon landing?")',
         "description": "Search Wikipedia and get the information about the first moon landing."},
    ],
}

# Official prompt (wikipedia_search/tool.py, select_relevant_queries).
SELECT = """You are an expert AI assistant. Your task is to identify and select the most relevant queries from a list of Wikipedia search results that are most likely to address the user’s original question.

## Input

Original Query: `{query}`
Query Candidates from Wikipedia Search: `{candidates}`

## Instructions

1. Carefully read the original query and the list of query candidates.
2. Select the query candidates that are most relevant to the original query — i.e., those most likely to contain the information needed to answer the question.
3. Return the most relevant queries. If you think multiple queries are helpful, you can return up to 3 queries.
4. Return your output in the following format:

```
- Matched Queries: <list of matched queries>
- Matched Query IDs: <list of matched query ids>. Please make sure the ids are integers. And do not return empty list.
```

## Examples

Original Query: What is the capital of France?
Query Candidates from Wikipedia Search:
0. Closed-ended question
1. France
2. What Is a Nation?
3. Capital city
4. London
5. WhatsApp
6. French Revolution
7. Communes of France
8. Capital punishment
9. Louis XIV

Output:
- Matched Queries: France
- Matched Query IDs: [1]


Original Query: What is the mass of the moon?
Query Candidates from Wikipedia Search:
0. Moon
1. Planetary-mass moon
2. What If the Moon Didn't Exist
3. Earth mass
4. Moon landing
5. Mass
6. Colonization of the Moon
7. Planetary mass
8. Hollow Moon
9. Gravitation of the Moon

Output:
- Matched Queries: Moon, Planetary-mass moon
- Matched Query IDs: [0, 1]
"""

# Official prompt (web_search/tool.py, SUMMARIZE_PROMPT_TEMPLATE).
SUMMARIZE = """You are an expert AI assistant. Your task is to provide a clear, concise, and accurate answer to the user's query based **exclusively** on the provided reference information.

## Step-by-Step Instructions
1.  **Analyze the Query:** First, fully understand the user's query and identify the specific information being asked for.
2.  **Scan for Relevance:** Read through each numbered chunk in the reference information. Identify all chunks that contain information directly relevant to answering the query. A simple keyword match is not sufficient; the chunk must contain a substantive fact that helps answer the question.
3.  **Extract Key Facts & Synthesize:** From the relevant chunks, extract only the key facts and figures needed. Synthesize these extracted facts into a comprehensive, single-paragraph answer. Write the answer in your own words. **Do not** copy entire chunks.

## Output Format and Example

**IMPORTANT:** You must follow this format exactly.

### Example Input
- **User Query:** What were the key financial results for Q4 2023?
- **Reference Information:**
[1] The company's new "Project Starlight" initiative launched in January 2024.
[2] In Q4 2023, the company reported a total revenue of $5.2 million and a net profit of $800,000. This was a 15% increase in revenue compared to Q3 2023.
[3] Marketing spend in Q4 2023 was focused on digital channels, totaling $450,000.
[4] The CEO stated that the strong Q4 performance was driven by robust sales in the North American market.

### Example Output
Answer:
In the fourth quarter of 2023, the company achieved a total revenue of $5.2 million, which represented a 15% increase from the previous quarter, and a net profit of $800,000. The strong performance was attributed to robust sales in the North American market. The marketing expenditure for this period was $450,000.

---
## Your Turn

### User Query
{query}

### Reference Information
{reference}

### Output
"""

IDS = re.compile(r"Matched Query IDs\W*\[([^\]]*)\]")


def api(**params):
    """One MediaWiki API call; a network error gives None, like the official tool's per-page fallback."""
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=20) as r:
            return json.load(r)
    except OSError:
        return None


@lru_cache(maxsize=None)
def search(query):
    """[(title, abstract)] of Wikipedia's 10 best matches."""
    reply = api(action="query", list="search", srsearch=query, srlimit=10)
    hits = reply["query"]["search"] if reply and "query" in reply else []
    return tuple((h["title"], html.unescape(re.sub(r"<[^>]+>", "", h.get("snippet", "")))) for h in hits)


@lru_cache(maxsize=None)
def page_text(title):
    reply = api(action="query", prop="extracts", explaintext=1, redirects=1, titles=title)
    pages = reply["query"]["pages"] if reply and "query" in reply else []
    return pages[0].get("extract", "") if pages else ""


def chunks(text, size=200, overlap=20):
    """The official sliding window over words (web_search/tool.py, _chunk_website_content)."""
    words, out, start = text.split(), [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def url(title):
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


class LiveWikipedia:
    metadata = METADATA

    def __init__(self, device=None, workers=8, top_k=10):
        from transformers import AutoModel, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device, self.workers, self.top_k = device, workers, top_k
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.encoder = AutoModel.from_pretrained(ENCODER, dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def embed(self, texts, prefix):
        out = []
        for i in range(0, len(texts), 256):
            batch = self.tokenizer([prefix + t for t in texts[i:i + 256]], max_length=512, padding=True,
                                   truncation=True, return_tensors="pt").to(self.device)
            hidden = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.append(F.normalize(((hidden * mask).sum(1) / mask.sum(1)).float(), dim=-1))
        return torch.cat(out) if out else torch.zeros(0, 768, device=self.device)

    def run(self, queries, llm, tokenizer):
        """The tool's result (a dict) for each query; the LLM calls of all queries run as one batch each."""
        from agentflow import chat, generate
        with ThreadPoolExecutor(self.workers) as pool:
            hits = list(pool.map(search, queries))
        prompts = [chat(tokenizer, SELECT.format(query=q, candidates="\n".join(f"{i}. {t}" for i, (t, _) in
                                                                                enumerate(h))))
                   for q, h in zip(queries, hits)]
        picked = []
        for out, h in zip(generate(llm, prompts, 256), hits):
            ids = IDS.findall(out.outputs[0].text)
            ids = [int(i) for i in re.findall(r"\d+", ids[-1])] if ids else []
            picked.append(list(dict.fromkeys(i for i in ids if i < len(h)))[:3])
        jobs = [(q, h[i][0]) for q, h, ids in zip(range(len(queries)), hits, picked) for i in ids]
        with ThreadPoolExecutor(self.workers) as pool:
            texts = list(pool.map(page_text, [title for _, title in jobs]))
        pieces = [chunks(t) for t in texts]
        query_vectors = self.embed(queries, "query: ")
        chunk_vectors = self.embed([c for p in pieces for c in p], "passage: ")
        references, start = [], 0
        for (q, _), p in zip(jobs, pieces):
            scores = chunk_vectors[start:start + len(p)] @ query_vectors[q]
            start += len(p)
            best = scores.topk(min(self.top_k, len(p))).indices.tolist() if p else []
            references.append("\n".join(f"Chunk [{k + 1}]\n{p[j]}" for k, j in enumerate(best)))
        prompts = [chat(tokenizer, SUMMARIZE.format(query=queries[q], reference=ref)) for (q, _), ref in
                   zip(jobs, references)]
        answers = [o.outputs[0].text for o in generate(llm, prompts, 1024)] if prompts else []
        results = [{"query": q, "relevant_pages (to the query)": [], "other_pages (may be irrelevant to the query)": []}
                   for q in queries]
        retrieved = {(q, title): a if ref else None for (q, title), ref, a in zip(jobs, references, answers)}
        for q, (h, ids) in enumerate(zip(hits, picked)):
            for i, (title, abstract) in enumerate(h):
                page = {"title": title, "url": url(title), "abstract": abstract}
                if i in ids:
                    results[q]["relevant_pages (to the query)"].append({**page, "retrieved_information":
                                                                        retrieved[q, title]})
                else:
                    results[q]["other_pages (may be irrelevant to the query)"].append(page)
        return results

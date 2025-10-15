#In VS Code: Ctrl+Shift+P → Python: Select Interpreter → choose
#C:\Users\Pamela Bayona\Documents\Python\TPRD rag\.venv\Scripts\python.exe

#Fresh install 
# open cmd: cd "C:\Users\Pamela Bayona\Documents\Python\ai_agent" 
# #py -3 -m venv .venv 
# #.\.venv\Scripts\activate #python -m pip install -r requirements.txt 
#type the below in cmd to install the packages; NOT IN PYTHON! 
#python -m pip install -U langgraph langchain langchain-openai openai #pip install openai
#pip install openai

#Generate requirements file
#pip freeze > requirements.txt

#run on stream
#.\.venv\Scripts\activate
# streamlit run wto_app.py

from langgraph.func import entrypoint, task
from openai import OpenAI
import os, re

# 1) API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2) Minimal LLM wrapper
def llm(prompt: str, api_key: str | None = OPENAI_API_KEY, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a computational linguist and professional translator."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=900,
        temperature=0.3,
    )
    return resp.choices[0].message.content

# 3) Lightweight stats (pure Python)
def basic_stats(text: str):
    toks = re.findall(r"\b\w+\b", text.lower())
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    ttr = (len(set(toks)) / len(toks)) if toks else 0.0
    avg_len = (sum(len(re.findall(r'\b\w+\b', s)) for s in sents) / len(sents)) if sents else 0.0
    return {
        "tokens": len(toks),
        "sentences": len(sents),
        "avg_sentence_len": round(avg_len, 2),
        "ttr": round(ttr, 3),
    }

def jaccard_token_overlap(a: str, b: str) -> float:
    A = set(re.findall(r"\b\w+\b", a.lower()))
    B = set(re.findall(r"\b\w+\b", b.lower()))
    return round((len(A & B) / len(A | B)) if (A or B) else 0.0, 3)

# 4) Tasks
@task
def analyze_source_task(text: str, domain: str) -> str:
    return llm(
        f"""Analyze the SOURCE text for domain '{domain}'.
Return compact Markdown with:
- **Language detected**
- **Key terms** (≤10) with part-of-speech tags
- **Named entities** (type: PER/ORG/LOC/DATE/etc.)
- **Style notes** (register, voice, hedging), and potential **translation pitfalls**"""
        + "\n\nSOURCE:\n" + text
    )

@task
def evaluate_translation_task(source: str, translation: str, domain: str) -> str:
    return llm(
        f"""Evaluate the TRANSLATION against the SOURCE for domain '{domain}'.
Return Markdown:
- **Adequacy** (0–5) and **Fluency** (0–5) with one-line justifications each
- **Terminology consistency**: a 5-row Markdown table:
| Term | Source Span | Translation Span | OK? | Note |
- **Register/style issues** (bullets)
- **Suggested improvements** (bullets, concise)
"""
        + f"\n\nSOURCE:\n{source}\n\nTRANSLATION:\n{translation}"
    )

@task
def polish_report_task(markdown: str) -> str:
    return llm(
        "Polish this report for clarity and succinctness. Preserve headings and tables. Return Markdown only.\n\n"
        + markdown
    )

# 5) Entrypoint (accepts dict or string)
@entrypoint()
def prompt_chaining_workflow(payload) -> str:
    # Back-compat: allow plain string (treated as source)
    if isinstance(payload, str):
        payload = {"source": payload, "translation": "", "domain": "general"}

    source = (payload.get("source") or "").strip()
    translation = (payload.get("translation") or "").strip()
    domain = (payload.get("domain") or "general").strip()

    if not source:
        return "Please provide source text."

    stats = basic_stats(source)
    header = (
        "# Bilingual Term & Quality Radar\n\n"
        f"**Domain:** {domain}\n\n"
        f"**Basic stats:** tokens={stats['tokens']}, "
        f"sentences={stats['sentences']}, "
        f"avg_sentence_len={stats['avg_sentence_len']}, "
        f"TTR={stats['ttr']}\n"
    )

    src_md = analyze_source_task(source, domain).result()

    if translation:
        overlap = jaccard_token_overlap(source, translation)
        eval_md = evaluate_translation_task(source, translation, domain).result()
        combined = (
            f"{header}\n## Source analysis\n{src_md}\n\n"
            f"## Quick overlap\nEstimated token-set Jaccard overlap: **{overlap}**\n\n"
            f"## Translation review\n{eval_md}\n"
        )
    else:
        combined = (
            f"{header}\n## Source analysis\n{src_md}\n\n"
            "## Next step suggestion\n"
            "Provide a translation or ask for a bilingual term bank to be generated."
        )

    return polish_report_task(combined).result()

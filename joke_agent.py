#In VS Code: Ctrl+Shift+P → Python: Select Interpreter → choose
#C:\Users\Pamela Bayona\Documents\Python\TPRD rag\.venv\Scripts\python.exe

#Fresh install 
# open cmd: cd "C:\Users\Pamela Bayona\Documents\Python\ai_agent" #py -3 -m venv .venv #.\.venv\Scripts\activate #python -m pip install -r requirements.txt 
#type the below in cmd to install the packages; NOT IN PYTHON! 
#python -m pip install -U langgraph langchain langchain-openai openai #pip install openai
#pip install openai

#Generate requirements file
#pip freeze > requirements.txt

#run on stream
#.\.venv\Scripts\activate
# streamlit run joke_app.py

from langgraph.func import entrypoint, task
from openai import OpenAI
import os

# Read your key from env (recommended) or pass explicitly to llm(...)
#OPENAI_API_KEY = os.getenv("73667704832f49c5beb884fa7106a300")  # set this in your shell
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def llm(prompt: str, api_key: str | None = OPENAI_API_KEY, model: str = "gpt-4") -> str:
    """Thin wrapper around OpenAI Chat Completions that returns text."""
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a witty comedian."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=750,
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Let the workflow surface the error text; alternatively raise.
        return f"Error generating speech: {e}"

# Tasks
@task
def generate_joke(topic: str) -> str:
    """First LLM call to generate initial joke"""
    return llm(f"Write a short joke about {topic}")

def check_punchline(joke: str) -> str:
    """Gate function to check if the joke has a punchline"""
    # Simple check - does the joke contain "?" or "!"
    if "?" in joke or "!" in joke:
        return "Fail"
    return "Pass"

@task
def improve_joke(joke: str) -> str:
    """Second LLM call to improve the joke"""
    return llm(f"Make this joke funnier by adding wordplay: {joke}")

@task
def polish_joke(joke: str) -> str:
    """Third LLM call for final polish"""
    return llm(f"Add a surprising twist to this joke: {joke}")

@entrypoint()
def prompt_chaining_workflow(topic: str) -> str:
    original_joke = generate_joke(topic).result()
    if check_punchline(original_joke) == "Pass":
        return original_joke
    improved_joke = improve_joke(original_joke).result()
    return polish_joke(improved_joke).result()

# Invoke
for step in prompt_chaining_workflow.stream("UNCTAD's Ministerial conference and TRUMP", stream_mode="updates"):
    print(step)
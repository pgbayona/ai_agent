# app.py
import os
import streamlit as st

# Import your LangGraph entrypoint from your agent file
# If your file isn't named agent.py, change the import below accordingly.
from joke_agent import prompt_chaining_workflow

st.set_page_config(page_title="Joke Agent", page_icon="🎭")
st.title("🎭 Joke Generator (LangGraph + OpenAI)")

# --- API key input (only if not already set in environment) ---
if not os.getenv("OPENAI_API_KEY"):
    with st.expander("🔑 Set OpenAI API key for this session"):
        key_input = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        if key_input:
            os.environ["OPENAI_API_KEY"] = key_input
            st.success("API key set for this session.")

# --- Topic input ---
topic = st.text_input("Topic", placeholder="e.g., cats, espresso, Kubernetes")

# Optional: show intermediate steps
show_steps = st.checkbox("Show workflow updates (stream)", value=False)

# --- Generate button ---
if st.button("Generate joke"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide your OpenAI API key above or set OPENAI_API_KEY in your environment.")
    elif not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Cooking up something funny..."):
            try:
                if show_steps:
                    # Stream updates as they come
                    steps_box = st.container()
                    final = None
                    for step in prompt_chaining_workflow.stream(topic.strip(), stream_mode="updates"):
                        with steps_box:
                            st.write(step)  # shows each update dict/string
                        final = step
                    # Best-effort final extraction
                    joke = final if isinstance(final, str) else str(final)
                else:
                    # Simpler: single-shot result
                    result = prompt_chaining_workflow.invoke(topic.strip())
                    joke = result if isinstance(result, str) else str(result)

                st.subheader("Final joke")
                st.write(joke)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
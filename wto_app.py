# app.py
import os
import streamlit as st
from wto_agent import prompt_chaining_workflow

st.set_page_config(page_title="Bilingual Term & Quality Radar", page_icon="🧭")
st.title("🧭 Bilingual Term & Quality Radar")

if not os.getenv("OPENAI_API_KEY"):
    with st.expander("🔑 Set OpenAI API key for this session"):
        key_input = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        if key_input:
            os.environ["OPENAI_API_KEY"] = key_input
            st.success("API key set for this session.")

domain = st.selectbox(
    "Domain",
    ["general", "legal", "trade policy", "technical", "news", "academic"],
    index=2,
)

source = st.text_area("Source text", height=220, placeholder="Paste source text here…")
translation = st.text_area("Translation (optional)", height=220, placeholder="Paste translation to review…")

show_steps = st.checkbox("Show workflow updates (stream)", value=False)

if st.button("Analyze"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide your OpenAI API key above or set OPENAI_API_KEY in your environment.")
    elif not source.strip():
        st.warning("Please paste source text.")
    else:
        payload = {"source": source.strip(), "translation": translation.strip(), "domain": domain}
        with st.spinner("Analyzing…"):
            try:
                if show_steps:
                    steps_box = st.container()
                    final = None
                    for step in prompt_chaining_workflow.stream(payload, stream_mode="updates"):
                        with steps_box:
                            st.write(step)
                        final = step
                    result = final if isinstance(final, str) else str(final)
                else:
                    result = prompt_chaining_workflow.invoke(payload)

                st.markdown("### Report")
                st.markdown(result)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
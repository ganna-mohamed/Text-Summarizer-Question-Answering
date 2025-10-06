import streamlit as st
from transformers import pipeline

# ---- لازم يكون ده أول Streamlit command ----
st.set_page_config(page_title="Text Summarizer & Q&A", layout="wide")

# =========================
# 1. Load Models
# =========================
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", framework="pt")
    return summarizer, qa_pipeline

summarizer, qa_pipeline = load_pipelines()

# =========================
# 2. Streamlit UI
# =========================
st.title("📝 Text Summarization & Question Answering")

col1, col2 = st.columns(2)

# ---- العمود الأول: التلخيص ----
with col1:
    st.subheader("Step 1: Enter your text")
    context = st.text_area("Paste a paragraph or article here:", height=200)

    summary_text = ""
    if st.button("Summarize Text", key="summarize"):
        if context.strip():
            with st.spinner("Summarizing..."):
                summary = summarizer(context, max_length=100, min_length=30, do_sample=False)
                summary_text = summary[0]['summary_text']
            st.success("✅ Done!")
            st.write("### Summary:")
            st.info(summary_text)
        else:
            st.warning("⚠ Please enter some text first.")

# ---- العمود الثاني: الأسئلة والإجابات ----
with col2:
    st.subheader("Step 2: Ask a question about the text (or the summary)")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer", key="qa"):
        if question.strip():
            with st.spinner("Searching for the answer..."):
                input_context = summary_text if summary_text else context
                if input_context.strip():
                    result = qa_pipeline(question=question, context=input_context)
                    st.success("✅ Answer Found!")
                    st.write("*Answer:*", result["answer"])
                    st.caption(f"Confidence: {result['score']:.3f}")
                else:
                    st.warning("⚠ Please provide text or summary first.")
        else:
            st.warning("⚠ Please enter a question.")
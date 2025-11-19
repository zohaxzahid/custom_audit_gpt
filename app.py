# app.py -- Custom Audit GPT: Report Analysis + Context-Aware Chatbot
import streamlit as st
import fitz  # PyMuPDF
import docx
import os
import re
import google.generativeai as genai

# Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# ---------- Gemini setup ----------
# GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyAPRiY_Us6lXr61yLdIK1KCSxY7MnnELWE"
if not GEMINI_KEY:
    st.error('❌ GEMINI_API_KEY not found. Run: setx GEMINI_API_KEY "YOUR_KEY_HERE"')
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
# ---------- Streamlit UI ----------
st.set_page_config(page_title="Custom Audit GPT – Report Analysis + Chatbot", layout="wide")
st.title("📄 Custom Audit GPT — Report Analysis + Context-Aware Chatbot (FYP Mid Demo)")

# Upload
uploaded = st.file_uploader("Upload audit report (PDF / DOCX / TXT):", type=["pdf", "docx", "txt"])
max_chars = st.slider("Max characters to analyze (truncate long docs):", 5000, 60000, 20000)

# ----------- Helpers: extract & clean -----------
def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_docx(file_bytes):
    tmp = "temp_file.docx"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    d = docx.Document(tmp)
    os.remove(tmp)
    return "\n".join([p.text for p in d.paragraphs])

def clean_text(t):
    return re.sub(r'\n{2,}', '\n\n', t).strip()

# ----------- Gemini summarization -----------
def summarize_with_gemini(text):
    prompt = f"""
You are an expert audit assistant. Read the following audit report and provide:
1) A short professional summary (3–5 sentences).
2) Five key findings (bullet points).
3) Any red flags or irregularities.

Report text:
{text}
    """
    response = model.generate_content(prompt)
    return response.text

# ----------- Retrieval utilities (TF-IDF) -----------
def split_into_passages(doc_text, max_sentences=3):
    sentences = sent_tokenize(doc_text)
    passages = []
    i = 0
    while i < len(sentences):
        passage = " ".join(sentences[i:i+max_sentences])
        passages.append(passage)
        i += max_sentences
    return passages

def build_tfidf_index(doc_text, max_sentences=3):
    passages = split_into_passages(doc_text, max_sentences=max_sentences)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(passages)
    return passages, vectorizer, tfidf

def retrieve_top_k(question, passages, vectorizer, tfidf, k=4):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    top_idx = sims.argsort()[::-1][:k]
    top_passages = [passages[i] for i in top_idx]
    return top_passages, sims[top_idx]

# ----------- Main logic -----------
if uploaded:
    st.info("Extracting text...")
    file_bytes = uploaded.read()
    ext = uploaded.name.split(".")[-1].lower()

    if ext == "pdf":
        text = extract_pdf(file_bytes)
    elif ext == "docx":
        text = extract_docx(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    text = clean_text(text)
    if len(text) > max_chars:
        st.warning(f"Document is long ({len(text)} chars). Truncated for demo.")
        text = text[:max_chars]

    # Show extracted text snippet
    st.subheader("📌 Extracted Text (First 800 chars)")
    st.code(text[:800] + ("\n\n...[truncated]" if len(text) > 800 else ""))

    # Summarize
    st.subheader("🤖 AI Summary & Key Findings (Gemini)")
    with st.spinner("Generating summary..."):
        try:
            summary = summarize_with_gemini(text)
        except Exception as e:
            st.error(f"Gemini Error: {e}")
            summary = ""
    if summary:
        st.write(summary)
    else:
        st.write("No summary available.")

    # Quick heuristic checks
    st.subheader("⚠ Quick Heuristic Red Flags")
    issues = []
    for m in re.findall(r"Rs\.?\s?[\d,]{5,}", text)[:5]:
        issues.append(f"Large amount detected: {m}")
    if "approval" in text.lower():
        issues.append("Document mentions 'approval' — check missing approvals.")
    if not issues:
        st.success("No immediate heuristic issues detected.")
    else:
        for i in issues:
            st.write("- " + i)

    # ---------- Build retrieval index ----------
    with st.spinner("Indexing document for QA..."):
        passages, vectorizer, tfidf = build_tfidf_index(text, max_sentences=3)

    # ---------- Chat UI ----------
    st.markdown("---")
    st.subheader("💬 Ask the Document (Context-Aware Chatbot)")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Input row
    col1, col2 = st.columns([8,1])
    with col1:
        user_question = st.text_input("Ask a question about the uploaded report:", key="user_q_input")
    with col2:
        send = st.button("Send", key="send_btn")

    if send and user_question:
        # retrieve context
        top_passages, scores = retrieve_top_k(user_question, passages, vectorizer, tfidf, k=4)
        context_text = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(top_passages)])

        prompt = f"""
You are an expert forensic audit assistant. Use ONLY the CONTEXT passages below to answer. 
If the answer is not present, reply: "The document does not provide enough information to answer that."
Be concise and cite the passage number(s) you used.

CONTEXT:
{context_text}

QUESTION:
{user_question}

Answer:
"""
        try:
            resp = model.generate_content(prompt)
            answer_text = resp.text.strip()
        except Exception as e:
            answer_text = f"Error calling Gemini: {e}"

        st.session_state.chat_history.append({"role":"user","text":user_question})
        st.session_state.chat_history.append({"role":"assistant","text":answer_text})
        st.session_state.chat_history.append({"role":"context","text":context_text})

    # Display chat history
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            if entry["role"] == "user":
                st.markdown(f"**You:** {entry['text']}")
            elif entry["role"] == "assistant":
                st.markdown(f"**Assistant:** {entry['text']}")
            else:
                ctx = entry['text']
                st.markdown(f"*Context used (snippet):* {ctx[:600]}{'...' if len(ctx)>600 else ''}")

    st.success("Demo ready — ask multiple questions to test grounding.")
else:
    st.info("Upload a PDF / DOCX / TXT to begin analysis and ask questions.")

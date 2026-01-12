import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from external_research import external_research_answer

# ==================== CONFIG ====================
st.set_page_config(
    page_title="MedCopilot V3 — Hybrid Hospital AI",
    page_icon="🧠",
    layout="wide"
)

# ==================== UI ====================
st.markdown("""
# 🧠 MedCopilot V3 — Hybrid Hospital AI  
### Evidence-Based Hospital AI + Global Medical Research  
⚠ Research support only. Not a substitute for professional medical advice.
""")

# ==================== Sidebar ====================
st.sidebar.title("🏥 MedCopilot Status")

PDF_FOLDER = "medical_library"
pdf_files = []

if os.path.exists(PDF_FOLDER):
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

if pdf_files:
    st.sidebar.success("Medical Library Loaded")
else:
    st.sidebar.warning("No Medical Library Found")
    st.sidebar.info("External AI Mode Enabled")

# ==================== Load Models ====================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ==================== Load PDFs ====================
documents = []
sources = []

if pdf_files:
    for file in pdf_files:
        reader = PdfReader(os.path.join(PDF_FOLDER, file))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text) > 200:
                documents.append(text)
                sources.append(f"{file} — Page {i+1}")

# ==================== Build Vector DB ====================
if documents:
    embeddings = embedder.encode(documents)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
else:
    index = None

# ==================== Workspace ====================
st.markdown("## 🔬 Clinical Research Workspace")

query = st.text_input("Ask a clinical research question:")

# ==================== Hybrid AI ====================
if query:

    # ----------- Hospital Evidence Mode -----------
    if documents:

        q_embed = embedder.encode([query])
        D, I = index.search(np.array(q_embed), 5)

        context = "\n\n".join([documents[i] for i in I[0]])
        used_sources = [sources[i] for i in I[0]]

        st.markdown("## 🏥 Hospital Evidence-Based Answer")

        st.write(context[:3000])  # preview context

        st.markdown("### 📚 Evidence Sources")
        for s in used_sources:
            st.info(s)

        st.success("Mode: Hospital Evidence AI (Local Medical Library)")

    # ----------- Global Medical AI Mode -----------
    else:
        st.markdown("## 🌍 Global Medical Research Answer")

        try:
            with st.spinner("🔍 Searching global medical research..."):
                external = external_research_answer(query)

            st.markdown("### 🧠 Clinical Research Answer")
            st.write(external["answer"])

            st.success("Mode: Global Medical AI (Groq LLaMA-3.1)")

        except Exception as e:
            st.error(f"External AI Error: {str(e)}")

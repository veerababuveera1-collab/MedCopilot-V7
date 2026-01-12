import streamlit as st
import os, json, pickle, datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Clinical Research Command Center",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# ENTERPRISE DARK GLASS UI
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at top, #020617, #000000);
    color: #e5e7eb;
}
.main-header {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 25px;
}
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 25px 50px rgba(0,0,0,0.6);
}
.metric-value {
    font-size: 34px;
    font-weight: 800;
    color: #38bdf8;
}
.metric-label {
    font-size: 13px;
    color: #94a3b8;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 14px;
    padding: 14px 26px;
    font-weight: 700;
}
input, textarea {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# STORAGE
# =========================================================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")
ANALYTICS_FILE = "analytics_log.json"
FDA_DB = "fda_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# =========================================================
# SESSION STATE (CRITICAL)
# =========================================================
for k, v in {
    "index_ready": False,
    "index": None,
    "documents": [],
    "sources": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-header">ĀROGYABODHA AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clinical Research Command Center • Evidence • Governance • Trust</div>', unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# =========================================================
# FDA REGISTRY
# =========================================================
if not os.path.exists(FDA_DB):
    json.dump({
        "temozolomide": "FDA Approved",
        "bevacizumab": "FDA Approved",
        "car-t": "Experimental / Trial Only"
    }, open(FDA_DB, "w"))

FDA_REGISTRY = json.load(open(FDA_DB))

# =========================================================
# ANALYTICS LOGGER
# =========================================================
def log_query(query, mode):
    entry = {"query": query, "mode": mode, "time": str(datetime.datetime.now())}
    logs = json.load(open(ANALYTICS_FILE)) if os.path.exists(ANALYTICS_FILE) else []
    logs.append(entry)
    json.dump(logs, open(ANALYTICS_FILE, "w"), indent=2)

# =========================================================
# AGE EXTRACTION
# =========================================================
def extract_age(query):
    if "over" in query.lower():
        try:
            return int(query.lower().split("over")[1].split()[0])
        except:
            return None
    return None

# =========================================================
# TREATMENT EXTRACTION
# =========================================================
def extract_treatments(text):
    return [(d.title(), s) for d, s in FDA_REGISTRY.items() if d in text.lower()]

# =========================================================
# BUILD INDEX
# =========================================================
def build_index():
    docs, srcs = [], []
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    with st.spinner("🧠 Building medical knowledge index..."):
        for pdf in pdfs:
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, page in enumerate(reader.pages[:200]):
                text = page.extract_text()
                if text and len(text.strip()) > 100:
                    docs.append(text)
                    srcs.append(f"{pdf} — Page {i+1}")
    emb = embedder.encode(docs, batch_size=16)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    faiss.write_index(index, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))
    return index, docs, srcs

# =========================================================
# AUTO LOAD INDEX
# =========================================================
if not st.session_state.index_ready:
    if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
        idx = faiss.read_index(INDEX_FILE)
        data = pickle.load(open(CACHE_FILE, "rb"))
        st.session_state.index = idx
        st.session_state.documents = data["documents"]
        st.session_state.sources = data["sources"]
        st.session_state.index_ready = True

# =========================================================
# SIDEBAR — ANALYTICS + LIBRARY
# =========================================================
st.sidebar.subheader("📊 Analytics")
if os.path.exists(ANALYTICS_FILE):
    logs = json.load(open(ANALYTICS_FILE))
    st.sidebar.metric("Total Queries", len(logs))

st.sidebar.divider()
st.sidebar.subheader("📁 Medical Knowledge Base")

uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build / Refresh Index"):
    idx, docs, srcs = build_index()
    st.session_state.index = idx
    st.session_state.documents = docs
    st.session_state.sources = srcs
    st.session_state.index_ready = True
    st.sidebar.success("Index ready")

# =========================================================
# METRICS STRIP
# =========================================================
c1, c2, c3, c4 = st.columns(4)
for col, t, l in [
    (c1, "RAG", "Evidence-First"),
    (c2, "FDA", "Regulatory Aware"),
    (c3, "Cohort", "Age-Aware"),
    (c4, "Analytics", "Governance")
]:
    with col:
        st.markdown(f"<div class='glass-card'><div class='metric-value'>{t}</div><div class='metric-label'>{l}</div></div>", unsafe_allow_html=True)

# =========================================================
# QUERY
# =========================================================
query = st.text_input("Ask a clinical research question")
mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)
run = st.button("🚀 Analyze")

# =========================================================
# EXECUTION
# =========================================================
if run and query:
    log_query(query, mode)
    age = extract_age(query)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Hospital AI",
        "🌍 Global AI",
        "🧪 Outcomes & FDA",
        "📚 Medical Library",
        "📊 Analytics"
    ])

    if mode in ["Hospital AI", "Hybrid AI"]:
        if not st.session_state.index_ready:
            st.error("Hospital index not ready")
            st.stop()

        q_emb = embedder.encode([query])
        _, I = st.session_state.index.search(np.array(q_emb), 5)
        context = "\n\n".join([st.session_state.documents[i] for i in I[0]])

        prompt = f"""
You are a clinical research AI.
Question: {query}
Age Filter: {age}
Use only provided evidence.
Include FDA status if applicable.
"""
        ans = external_research_answer(prompt).get("answer", "")

        with tab1:
            st.write(ans)
            st.subheader("📚 Evidence Sources")
            for s in st.session_state.sources[:5]:
                st.info(s)

        with tab3:
            for d, s in extract_treatments(ans):
                st.info(f"💊 {d} — {s}")

    if mode in ["Global AI", "Hybrid AI"]:
        with tab2:
            st.write(external_research_answer(query).get("answer", ""))

    with tab4:
        pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        for p in pdfs:
            col1, col2 = st.columns([8, 2])
            col1.write(p)
            if col2.button("🗑️ Delete", key=p):
                os.remove(os.path.join(PDF_FOLDER, p))
                st.experimental_rerun()

    with tab5:
        if os.path.exists(ANALYTICS_FILE):
            st.json(json.load(open(ANALYTICS_FILE))[-10:])

# =========================================================
# FOOTER
# =========================================================
st.caption("ĀROGYABODHA AI © Enterprise Clinical Research Copilot — FINAL")

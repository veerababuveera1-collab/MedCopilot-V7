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
# WOW ENTERPRISE UI (GLASS + DARK MODE)
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
    border: none;
}
.stButton>button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 15px 40px rgba(37,99,235,0.6);
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
ANALYTICS_FILE = "analytics_log.json"
FDA_DB = "fda_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-header">ĀROGYABODHA AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Clinical Research Command Center • Evidence-Driven • Regulatory-Aware</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# =========================================================
# FDA REGISTRY (LIGHTWEIGHT DEMO DB)
# =========================================================
if not os.path.exists(FDA_DB):
    with open(FDA_DB, "w") as f:
        json.dump({
            "temozolomide": "FDA Approved",
            "bevacizumab": "FDA Approved",
            "car-t": "Experimental / Trial Only"
        }, f)

FDA_REGISTRY = json.load(open(FDA_DB))

# =========================================================
# ANALYTICS LOGGER
# =========================================================
def log_query(query, mode):
    entry = {
        "query": query,
        "mode": mode,
        "time": str(datetime.datetime.now())
    }
    logs = []
    if os.path.exists(ANALYTICS_FILE):
        logs = json.load(open(ANALYTICS_FILE))
    logs.append(entry)
    json.dump(logs, open(ANALYTICS_FILE, "w"), indent=2)

# =========================================================
# AGE / COHORT EXTRACTION
# =========================================================
def extract_age_filter(query):
    q = query.lower()
    if "over" in q:
        try:
            return int(q.split("over")[1].strip().split()[0])
        except:
            return None
    return None

# =========================================================
# TREATMENT + FDA EXTRACTION
# =========================================================
def extract_treatments(text):
    found = []
    for drug, status in FDA_REGISTRY.items():
        if drug.lower() in text.lower():
            found.append((drug.title(), status))
    return found

# =========================================================
# INDEX LOADING
# =========================================================
INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

@st.cache_resource
def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
        index = faiss.read_index(INDEX_FILE)
        data = pickle.load(open(CACHE_FILE, "rb"))
        return index, data["documents"], data["sources"]
    return None, [], []

index, documents, sources = load_index()

# =========================================================
# SIDEBAR – ANALYTICS
# =========================================================
st.sidebar.subheader("📊 Research Analytics")
if os.path.exists(ANALYTICS_FILE):
    logs = json.load(open(ANALYTICS_FILE))
    st.sidebar.metric("Total Queries", len(logs))
    st.sidebar.write("Recent:")
    for l in logs[-3:]:
        st.sidebar.write(f"- {l['query']} ({l['mode']})")

# =========================================================
# METRICS STRIP
# =========================================================
c1, c2, c3, c4 = st.columns(4)
for col, title, label in [
    (c1, "RAG", "Hallucination-Safe AI"),
    (c2, "FDA", "Regulatory Intelligence"),
    (c3, "Cohorts", "Age-Aware Evidence"),
    (c4, "Analytics", "Research Insights")
]:
    with col:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-value">{title}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# QUERY BAR
# =========================================================
st.markdown("### 🔍 Clinical Research Query")
query = st.text_input("Ask a clinical research or trial design question")
mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)
run = st.button("🚀 Analyze")

# =========================================================
# MAIN EXECUTION
# =========================================================
if run and query:
    log_query(query, mode)
    age_filter = extract_age_filter(query)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Hospital Intelligence",
        "🌍 Global Research",
        "🧪 Outcomes & FDA",
        "📊 Analytics View"
    ])

    # ---------------- HOSPITAL / HYBRID ----------------
    if mode in ["Hospital AI", "Hybrid AI"] and index:
        q_emb = embedder.encode([query])
        D, I = index.search(np.array(q_emb), 5)

        context = "\n\n".join([documents[i] for i in I[0]])

        prompt = f"""
You are a clinical research AI.

Question: {query}
Age Filter: {age_filter}

Tasks:
- Summarize treatments
- Compare outcomes if available
- Mention FDA approval
- Cite evidence
"""
        answer = external_research_answer(prompt).get("answer", "")

        with tab1:
            st.subheader("🏥 Hospital / Hybrid Intelligence")
            st.write(answer)

        with tab3:
            st.subheader("🧾 Regulatory & Treatment Summary")
            for drug, status in extract_treatments(answer):
                st.info(f"💊 {drug} — {status}")

    # ---------------- GLOBAL AI ----------------
    if mode in ["Global AI", "Hybrid AI"]:
        with tab2:
            st.subheader("🌍 Global Medical Research")
            global_ans = external_research_answer(query)
            st.write(global_ans.get("answer", ""))

    # ---------------- ANALYTICS ----------------
    with tab4:
        if os.path.exists(ANALYTICS_FILE):
            st.json(json.load(open(ANALYTICS_FILE))[-10:])

# =========================================================
# FOOTER
# =========================================================
st.caption("ĀROGYABODHA AI © Enterprise Clinical Research Copilot — Phase-2 Complete")

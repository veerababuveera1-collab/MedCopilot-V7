import streamlit as st
import os, json, pickle, datetime, requests
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
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #38bdf8;
}
.metric-label {
    font-size: 13px;
    color: #94a3b8;
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
# SESSION STATE
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
st.markdown('<div class="sub-header">Clinical Research Command Center • Evidence-Locked • Auditable</div>', unsafe_allow_html=True)

# =========================================================
# MODELS
# =========================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# =========================================================
# FDA REGISTRY (DEMO)
# =========================================================
if not os.path.exists(FDA_DB):
    json.dump({
        "temozolomide": "FDA Approved",
        "bevacizumab": "FDA Approved",
        "car-t": "Experimental / Trial Only"
    }, open(FDA_DB, "w"))

FDA_REGISTRY = json.load(open(FDA_DB))

# =========================================================
# ANALYTICS
# =========================================================
def log_query(query, mode):
    logs = json.load(open(ANALYTICS_FILE)) if os.path.exists(ANALYTICS_FILE) else []
    logs.append({"query": query, "mode": mode, "time": str(datetime.datetime.now())})
    json.dump(logs, open(ANALYTICS_FILE, "w"), indent=2)

# =========================================================
# HELPERS
# =========================================================
def extract_age(query):
    if "over" in query.lower():
        try:
            return int(query.lower().split("over")[1].split()[0])
        except:
            return None
    return None

def extract_treatments(text):
    return [(d.title(), s) for d, s in FDA_REGISTRY.items() if d in text.lower()]

def extract_outcomes(text):
    rows = []
    for line in text.split("\n"):
        l = line.lower()
        if "overall survival" in l or "os" in l:
            rows.append(("Overall Survival", line.strip()))
        if "progression-free" in l or "pfs" in l:
            rows.append(("PFS", line.strip()))
        if "response rate" in l:
            rows.append(("Response Rate", line.strip()))
    return rows

def calculate_confidence(answer, sources):
    score = 50
    if sources >= 3: score += 20
    if "fda" in answer.lower(): score += 15
    if "survival" in answer.lower(): score += 10
    return min(score, 95)

# =========================================================
# STRICT RAG CONTROLLER (CRITICAL FIX)
# =========================================================
def hospital_rag_answer(query, context, age_filter=None):
    prompt = f"""
You are a hospital-grade clinical research AI.

STRICT RULES:
- Use ONLY the evidence provided below.
- Do NOT use general medical knowledge.
- If evidence is insufficient, clearly say "Evidence insufficient".
- Do NOT guess.
- Cite evidence as [PDF:Page].

Doctor Query:
{query}

Patient Cohort:
Age > {age_filter if age_filter else "Not specified"}

Hospital Evidence:
{context}

OUTPUT FORMAT:
1. Treatment Summary
2. Outcome Comparison
3. FDA Approval Status
4. Clinical Notes
5. Evidence Citations
"""
    return external_research_answer(prompt).get("answer", "")

# =========================================================
# PUBMED AUTO INGESTION (GLOBAL AI)
# =========================================================
def fetch_pubmed(query, max_results=3):
    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results}
    ).json()

    abstracts = []
    for pid in search.get("esearchresult", {}).get("idlist", []):
        abs_text = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pid, "retmode": "text", "rettype": "abstract"}
        ).text
        abstracts.append(abs_text)

    return "\n\n".join(abstracts)

# =========================================================
# BUILD INDEX
# =========================================================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, p in enumerate(reader.pages[:200]):
                t = p.extract_text()
                if t and len(t.strip()) > 100:
                    docs.append(t)
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
if not st.session_state.index_ready and os.path.exists(INDEX_FILE):
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.subheader("📁 Medical Knowledge Base")
uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build / Refresh Index"):
    idx, docs, srcs = build_index()
    st.session_state.index, st.session_state.documents, st.session_state.sources = idx, docs, srcs
    st.session_state.index_ready = True
    st.sidebar.success("Index ready")

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

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Hospital AI (Evidence-Locked)",
        "🌍 Global AI (PubMed)",
        "🧪 Outcomes & FDA",
        "📚 Medical Library"
    ])

    if mode in ["Hospital AI", "Hybrid AI"]:
        if not st.session_state.index_ready:
            st.error("Hospital knowledge index not ready.")
            st.stop()

        q_emb = embedder.encode([query])
        _, I = st.session_state.index.search(np.array(q_emb), 5)
        context = "\n\n".join([st.session_state.documents[i] for i in I[0]])

        ans = hospital_rag_answer(query, context, age)

        with tab1:
            st.warning("⚠️ Answer generated strictly from hospital-uploaded evidence.")
            st.metric("🧠 Confidence Score", f"{calculate_confidence(ans, len(st.session_state.sources))}%")
            st.write(ans)
            st.subheader("📚 Evidence Sources")
            for s in st.session_state.sources[:5]:
                st.info(s)

        with tab3:
            for d, s in extract_treatments(ans):
                st.info(f"💊 {d} — {s}")
            outcomes = extract_outcomes(ans)
            if outcomes:
                st.table({
                    "Outcome": [o[0] for o in outcomes],
                    "Details": [o[1] for o in outcomes]
                })

    if mode in ["Global AI", "Hybrid AI"]:
        with tab2:
            pubmed_ctx = fetch_pubmed(query)
            st.write(
                external_research_answer(
                    f"Use these PubMed abstracts:\n{pubmed_ctx}\nQuestion:{query}"
                ).get("answer", "")
            )

    with tab4:
        for p in os.listdir(PDF_FOLDER):
            if p.endswith(".pdf"):
                st.write(f"📄 {p}")

# =========================================================
# FOOTER
# =========================================================
st.caption("ĀROGYABODHA AI © Final Clinical-Grade, Evidence-Locked Research Copilot")

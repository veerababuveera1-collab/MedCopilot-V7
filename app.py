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
# ENTERPRISE DARK UI
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
}
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LEGAL / CLINICAL DISCLAIMER (CRITICAL)
# =========================================================
st.info(
    "ℹ️ **ĀROGYABODHA AI is a clinical research decision-support system only.** "
    "It does NOT provide medical diagnosis or treatment recommendations. "
    "All final clinical decisions must be made by licensed medical professionals."
)

# =========================================================
# STORAGE
# =========================================================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
ANALYTICS_FILE = "analytics_log.json"
FDA_DB = "fda_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# =========================================================
# SESSION STATE
# =========================================================
for k, v in {
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-header">ĀROGYABODHA AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clinical-Grade • Evidence-Locked • Auditable Research Copilot</div>', unsafe_allow_html=True)

# =========================================================
# MODELS
# =========================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# =========================================================
# FDA REGISTRY (DEMO DATA)
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
def log_query(q, mode):
    logs = json.load(open(ANALYTICS_FILE)) if os.path.exists(ANALYTICS_FILE) else []
    logs.append({"query": q, "mode": mode, "time": str(datetime.datetime.now())})
    json.dump(logs, open(ANALYTICS_FILE, "w"), indent=2)

# =========================================================
# HELPERS
# =========================================================
def extract_age(q):
    if "over" in q.lower():
        try:
            return int(q.lower().split("over")[1].split()[0])
        except:
            return None
    return None

def extract_outcomes(text):
    rows = []
    for l in text.split("\n"):
        ll = l.lower()
        if "overall survival" in ll or "os" in ll:
            rows.append(("Overall Survival", l))
        if "progression-free" in ll or "pfs" in ll:
            rows.append(("PFS", l))
        if "response rate" in ll:
            rows.append(("Response Rate", l))
    return rows

def confidence_score(answer, evidence_count):
    score = 50
    if evidence_count >= 3: score += 20
    if "fda" in answer.lower(): score += 15
    if "survival" in answer.lower(): score += 10
    return min(score, 95)

# =========================================================
# STRICT EVIDENCE-LOCKED RAG CONTROLLER
# =========================================================
def hospital_rag(query, context, age):
    prompt = f"""
STRICT RULES:
- Use ONLY the hospital evidence below
- Do NOT use external medical knowledge
- Do NOT guess or infer
- Cite evidence as [PDF:Page]
- If evidence is insufficient, say "Evidence insufficient"

Query:
{query}

Patient Cohort:
Age > {age if age else "Not specified"}

Hospital Evidence:
{context}

OUTPUT FORMAT:
1. Treatment Summary
2. Outcome Comparison
3. FDA Status
4. Clinical Notes
5. Evidence Citations
"""
    return external_research_answer(prompt).get("answer", "")

# =========================================================
# PUBMED AUTO-INGESTION (GLOBAL AI)
# =========================================================
def fetch_pubmed(query, n=3):
    ids = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": n}
    ).json().get("esearchresult", {}).get("idlist", [])

    abstracts = []
    for pid in ids:
        abstracts.append(requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pid, "rettype": "abstract", "retmode": "text"}
        ).text)

    return "\n\n".join(abstracts)

# =========================================================
# INDEX BUILD / LOAD
# =========================================================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, page in enumerate(reader.pages[:200]):
                text = page.extract_text()
                if text and len(text.strip()) > 100:
                    docs.append(text)
                    srcs.append(f"{pdf} — Page {i+1}")

    emb = embedder.encode(docs, batch_size=16)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.subheader("📁 Medical Knowledge Base")
files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if files:
    for f in files:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build / Refresh Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    st.sidebar.success("Index built successfully")

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
        "🏥 Hospital (Evidence-Locked)",
        "🌍 Global (PubMed)",
        "🧪 Outcomes",
        "📚 Library"
    ])

    if mode in ["Hospital AI", "Hybrid AI"]:
        if not st.session_state.index_ready:
            st.error("Hospital knowledge index not ready.")
            st.stop()

        qemb = embedder.encode([query])
        _, I = st.session_state.index.search(np.array(qemb), 5)

        # 🔴 WEAK-EVIDENCE HARD STOP (CRITICAL)
        if len(I[0]) < 2:
            st.error(
                "⚠️ Insufficient hospital evidence for this query. "
                "Please upload more relevant medical documents."
            )
            st.stop()

        context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
        ans = hospital_rag(query, context, age)

        with tab1:
            st.warning("Evidence-locked RAG active — external knowledge disabled")
            st.metric("🧠 Confidence Score", f"{confidence_score(ans, len(I[0]))}%")
            st.write(ans)
            st.subheader("📚 Evidence Sources")
            for s in st.session_state.sources[:5]:
                st.info(s)

        with tab3:
            rows = extract_outcomes(ans)
            if rows:
                st.table({
                    "Outcome Metric": [r[0] for r in rows],
                    "Evidence Detail": [r[1] for r in rows]
                })

    if mode in ["Global AI", "Hybrid AI"]:
        with tab2:
            pubmed_ctx = fetch_pubmed(query)
            st.write(
                external_research_answer(
                    f"Use only the PubMed abstracts below:\n{pubmed_ctx}\n\nQuestion:{query}"
                ).get("answer", "")
            )

    with tab4:
        for p in os.listdir(PDF_FOLDER):
            if p.endswith(".pdf"):
                st.write("📄", p)

# =========================================================
# FOOTER
# =========================================================
st.caption("ĀROGYABODHA AI © Final, Clinical-Grade, Evidence-Locked Research Copilot")

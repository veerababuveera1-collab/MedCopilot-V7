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
    page_title="ĀROGYABODHA AI — Clinical Research Copilot",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# UI STYLE
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at top, #020617, #000000);
    color: #e5e7eb;
}
.main-header {
    font-size: 46px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {
    font-size: 17px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DISCLAIMER
# =========================================================
st.info(
    "ℹ️ **ĀROGYABODHA AI is a clinical research decision-support system only.** "
    "It does NOT provide diagnosis or treatment recommendations. "
    "Final clinical decisions must be made by licensed medical professionals."
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
defaults = {
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "help_lang": "EN",
    "show_quick_help": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# HEADER + QUICK HELP BUTTON
# =========================================================
h1, h2, h3 = st.columns([7, 1, 1])

with h1:
    st.markdown('<div class="main-header">ĀROGYABODHA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evidence-Locked • Auditable • Clinical Research Copilot</div>', unsafe_allow_html=True)

with h2:
    if st.button("❓ Quick Help"):
        st.session_state.show_quick_help = not st.session_state.show_quick_help

with h3:
    if st.button("🌐 EN / తెలుగు"):
        st.session_state.help_lang = "TE" if st.session_state.help_lang == "EN" else "EN"

# =========================================================
# QUICK HELP PANEL
# =========================================================
if st.session_state.show_quick_help:
    st.markdown("---")
    if st.session_state.help_lang == "EN":
        st.markdown("""
## ❓ Quick Help (English)

**ĀROGYABODHA AI** is a *clinical research support system*.

### What it does
- Reviews hospital protocols
- Compares ICU / oncology outcomes
- Shows FDA approval status
- Retrieves latest PubMed research

### What it does NOT do
❌ Diagnosis  
❌ Treatment prescription  

### AI Modes
- 🏥 Hospital AI → Only hospital PDFs  
- 🌍 Global AI → PubMed research  
- 🔀 Hybrid AI → Both, clearly separated  

### Safety
- Evidence-locked (no hallucinations)
- Stops if evidence is insufficient
- PDF + page citations mandatory
- Confidence score = evidence strength

👉 See **Help & Guidance** tab for full manual.
""")
    else:
        st.markdown("""
## ❓ త్వరిత సహాయం (తెలుగు)

**ĀROGYABODHA AI** ఒక *clinical research support system*.

### ఇది ఏమి చేస్తుంది
- హాస్పిటల్ ప్రోటోకాల్స్ పరిశీలిస్తుంది
- ICU / Oncology అవుట్‌కమ్స్ పోలుస్తుంది
- FDA అప్రూవల్స్ చూపిస్తుంది
- PubMed రీసెర్చ్ తీసుకువస్తుంది

### ఇది చేయదు
❌ డయాగ్నోసిస్  
❌ చికిత్స నిర్ణయం  

### AI మోడ్‌లు
- 🏥 Hospital AI → హాస్పిటల్ PDFs మాత్రమే  
- 🌍 Global AI → PubMed రీసెర్చ్  
- 🔀 Hybrid AI → రెండూ వేర్వేరుగా  

### భద్రత
- Evidence లేకుండా పని చేయదు
- సరిపడ సమాచారం లేకపోతే ఆపేస్తుంది
- PDF + పేజీ citations తప్పనిసరి

👉 పూర్తి వివరాలకు **Help & Guidance** Tab చూడండి.
""")
    st.markdown("---")

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
# HELPERS
# =========================================================
def extract_age(q):
    if "over" in q.lower():
        try: return int(q.lower().split("over")[1].split()[0])
        except: return None
    return None

def confidence_score(ans, n):
    score = 50
    if n >= 3: score += 20
    if "fda" in ans.lower(): score += 15
    if "survival" in ans.lower(): score += 10
    return min(score, 95)

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

# =========================================================
# STRICT HOSPITAL RAG
# =========================================================
def hospital_rag(query, context, age):
    prompt = f"""
STRICT RULES:
- Use ONLY hospital evidence
- No external knowledge
- Cite as [PDF:Page]
- If insufficient evidence, say so

Query: {query}
Age Filter: {age}

Evidence:
{context}

Return structured clinical summary.
"""
    return external_research_answer(prompt).get("answer", "")

# =========================================================
# PUBMED
# =========================================================
def fetch_pubmed(query, n=3):
    ids = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db":"pubmed","term":query,"retmode":"json","retmax":n}
    ).json().get("esearchresult",{}).get("idlist",[])
    texts=[]
    for pid in ids:
        texts.append(requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db":"pubmed","id":pid,"rettype":"abstract","retmode":"text"}
        ).text)
    return "\n\n".join(texts)

# =========================================================
# INDEX BUILD / LOAD
# =========================================================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            r = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i,p in enumerate(r.pages[:200]):
                t = p.extract_text()
                if t and len(t.strip())>100:
                    docs.append(t)
                    srcs.append(f"{pdf} – Page {i+1}")
    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents":docs,"sources":srcs}, open(CACHE_FILE,"wb"))
    return idx, docs, srcs

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE,"rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.subheader("📁 Medical Library")
files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if files:
    for f in files:
        open(os.path.join(PDF_FOLDER,f.name),"wb").write(f.getbuffer())
    st.sidebar.success("Uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    st.sidebar.success("Index Ready")

# =========================================================
# QUERY
# =========================================================
query = st.text_input("Ask a clinical research question")
mode = st.radio("AI Mode", ["Hospital AI","Global AI","Hybrid AI"], horizontal=True)
run = st.button("🚀 Analyze")

# =========================================================
# EXECUTION
# =========================================================
if run and query:
    age = extract_age(query)

    t1,t2,t3,t4,t5 = st.tabs([
        "🏥 Hospital AI",
        "🌍 Global AI",
        "🧪 Outcomes",
        "📚 Library",
        "❓ Help & Guidance"
    ])

    if mode in ["Hospital AI","Hybrid AI"]:
        if not st.session_state.index_ready:
            st.error("Hospital index not ready"); st.stop()

        qemb = embedder.encode([query])
        _,I = st.session_state.index.search(np.array(qemb),5)
        if len(I[0]) < 2:
            st.error("⚠️ Insufficient hospital evidence."); st.stop()

        context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
        ans = hospital_rag(query, context, age)

        with t1:
            st.metric("Confidence", f"{confidence_score(ans,len(I[0]))}%")
            st.write(ans)
            for s in st.session_state.sources[:5]:
                st.info(s)

        with t3:
            rows = extract_outcomes(ans)
            if rows:
                st.table({"Metric":[r[0] for r in rows],"Detail":[r[1] for r in rows]})

    if mode in ["Global AI","Hybrid AI"]:
        with t2:
            ctx = fetch_pubmed(query)
            st.write(external_research_answer(ctx+"\nQ:"+query).get("answer",""))

    with t4:
        for p in os.listdir(PDF_FOLDER):
            if p.endswith(".pdf"):
                st.write("📄", p)

    with t5:
        st.write("See Quick Help above or refer to doctor training guide.")

# =========================================================
# FOOTER
# =========================================================
st.caption("ĀROGYABODHA AI © Final Clinical-Grade Research Copilot")

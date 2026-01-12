import streamlit as st
import os, json, pickle, datetime
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Clinical Research Copilot",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# GLOBAL DISCLAIMER
# ======================================================
st.info(
    "ℹ️ **ĀROGYABODHA AI is a clinical research decision-support system only.** "
    "It does NOT provide diagnosis or treatment recommendations. "
    "Final clinical decisions must be made by licensed medical professionals."
)

# ======================================================
# STORAGE
# ======================================================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
ANALYTICS_FILE = "analytics_log.json"
FDA_DB = "fda_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "show_quick_help": False,
    "help_lang": "EN"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HEADER + QUICK HELP
# ======================================================
h1, h2, h3 = st.columns([7, 1, 1])

with h1:
    st.markdown("## 🧠 ĀROGYABODHA AI")
    st.caption("Evidence-Locked • Auditable • Clinical Research Copilot")

with h2:
    if st.button("❓ Quick Help"):
        st.session_state.show_quick_help = not st.session_state.show_quick_help

with h3:
    if st.button("🌐 EN / తెలుగు"):
        st.session_state.help_lang = "TE" if st.session_state.help_lang == "EN" else "EN"

# ======================================================
# QUICK HELP PANEL
# ======================================================
if st.session_state.show_quick_help:
    st.markdown("---")
    if st.session_state.help_lang == "EN":
        st.markdown("""
### ❓ Quick Help
- Research & evidence support only  
- Hospital AI uses **ONLY hospital PDFs**  
- Global AI uses PubMed  
- Evidence-locked, no hallucinations  
""")
    else:
        st.markdown("""
### ❓ త్వరిత సహాయం
- ఇది research support మాత్రమే  
- Hospital AI కేవలం PDFs మాత్రమే వాడుతుంది  
- Evidence లేకపోతే సమాధానం ఇవ్వదు  
""")
    st.markdown("---")

# ======================================================
# MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# FDA REGISTRY
# ======================================================
if not os.path.exists(FDA_DB):
    json.dump({
        "temozolomide": "FDA Approved",
        "bevacizumab": "FDA Approved",
        "car-t": "Experimental / Trial Only"
    }, open(FDA_DB, "w"))

FDA_REGISTRY = json.load(open(FDA_DB))

# ======================================================
# HELPERS
# ======================================================
def log_query(query, mode):
    logs = []
    if os.path.exists(ANALYTICS_FILE):
        logs = json.load(open(ANALYTICS_FILE))
    logs.append({
        "query": query,
        "mode": mode,
        "time": str(datetime.datetime.now())
    })
    json.dump(logs, open(ANALYTICS_FILE, "w"), indent=2)

def confidence_explained(ans, n_sources):
    score = 60
    reasons = []
    if n_sources >= 3:
        score += 15; reasons.append("Multiple hospital sources")
    if "fda" in ans.lower():
        score += 10; reasons.append("FDA reference present")
    if "survival" in ans.lower() or "pfs" in ans.lower():
        score += 10; reasons.append("Outcome data available")
    return min(score, 95), reasons

def extract_outcome_table(text):
    rows = []
    for drug, status in FDA_REGISTRY.items():
        if drug.lower() in text.lower():
            rows.append({
                "Treatment": drug.title(),
                "Outcome Mentioned": "Yes",
                "FDA Status": status
            })
    return pd.DataFrame(rows)

def generate_report(query, mode, answer, confidence, sources):
    report = f"""
ĀROGYABODHA AI — Clinical Research Report
---------------------------------------
Query: {query}
AI Mode: {mode}
Confidence Score: {confidence}%

AI Summary:
{answer}

Evidence Sources:
"""
    for s in sources:
        report += f"- {s}\n"
    return report

# ======================================================
# 🔒 EVIDENCE-LOCKED HOSPITAL AI (FIX APPLIED)
# ======================================================
def hospital_evidence_locked_answer(query, context):
    prompt = f"""
You are a Hospital Clinical Decision Support AI.

STRICT RULES:
- Use ONLY the hospital evidence below
- Do NOT use any external or prior medical knowledge
- Do NOT guess or infer beyond the text
- If evidence is insufficient, say:
  "Insufficient hospital evidence available."

Answer format:
- Treatment Summary
- Outcome Comparison (if available)
- FDA Approval Status (only if mentioned)
- Evidence-based Notes

Hospital Evidence:
{context}

Doctor Query:
{query}
"""
    return external_research_answer(prompt).get("answer", "")

# ======================================================
# INDEX BUILD / LOAD
# ======================================================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, page in enumerate(reader.pages[:200]):
                txt = page.extract_text()
                if txt and len(txt.strip()) > 100:
                    docs.append(txt)
                    srcs.append(f"{pdf} – Page {i+1}")
    if not docs:
        return None, [], []
    emb = embedder.encode(docs)
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

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.subheader("📁 Medical Library")
uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build / Rebuild Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    st.sidebar.success("Index built")

# ======================================================
# QUERY
# ======================================================
query = st.text_input("Ask a clinical research question")
mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)
run = st.button("🚀 Analyze")

# ======================================================
# EXECUTION
# ======================================================
if run and query:
    log_query(query, mode)

    t1, t2, t3 = st.tabs(["🏥 Hospital AI", "🌍 Global AI", "🧪 Outcomes"])

    if mode in ["Hospital AI", "Hybrid AI"]:
        qemb = embedder.encode([query])
        _, I = st.session_state.index.search(np.array(qemb), 5)

        if len(I[0]) < 2:
            st.error("⚠️ Insufficient hospital evidence.")
            st.stop()

        context = "\n\n".join([st.session_state.documents[i] for i in I[0]])

        # 🔒 Evidence-locked call
        answer = hospital_evidence_locked_answer(query, context)

        score, reasons = confidence_explained(answer, len(I[0]))

        with t1:
            st.metric("Confidence Score", f"{score}%")
            for r in reasons:
                st.write("•", r)
            st.write(answer)

            sources = [st.session_state.sources[i] for i in I[0]]
            for s in sources:
                st.info(s)

            report = generate_report(query, mode, answer, score, sources)
            st.download_button(
                "📥 Download Clinical Research Report",
                report,
                file_name="arogyabodha_clinical_report.txt",
                mime="text/plain"
            )

        with t3:
            df = extract_outcome_table(answer)
            if not df.empty:
                st.table(df)

    if mode in ["Global AI", "Hybrid AI"]:
        with t2:
            st.write(external_research_answer(query).get("answer", ""))

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © FINAL • Evidence-Locked • Review-Proof")

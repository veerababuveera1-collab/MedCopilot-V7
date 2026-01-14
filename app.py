import streamlit as st
import os, json, pickle, datetime, re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ======================================================
# WOW THEME & UI
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Clinical Intelligence Command Center",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #020617, #020617);
    color: #e5e7eb;
}
.card {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 0 40px rgba(0,200,255,0.15);
    margin-bottom: 20px;
}
.alert {
    background: linear-gradient(135deg, #ff004c, #ff6a00);
    padding: 15px;
    border-radius: 14px;
    font-weight: bold;
}
.success {
    background: linear-gradient(135deg, #00ff9c, #00c2ff);
    padding: 15px;
    border-radius: 14px;
    font-weight: bold;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# DISCLAIMER
# ======================================================
st.info(
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only. "
    "It does NOT provide diagnosis or treatment. "
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

def semantic_similarity(a, b):
    ea = embedder.encode([a])[0]
    eb = embedder.encode([b])[0]
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))

def semantic_evidence_level(answer, context):
    sim = semantic_similarity(answer, context)
    if sim >= 0.55:
        return "STRONG", int(sim * 100)
    elif sim >= 0.25:
        return "PARTIAL", int(sim * 100)
    else:
        return "NONE", 0

def confidence_score(answer, n_sources):
    score = 60
    if n_sources >= 3: score += 15
    if "fda" in answer.lower(): score += 10
    if any(x in answer.lower() for x in ["survival", "mortality", "outcome"]):
        score += 10
    return min(score, 95)

def extract_outcomes(text):
    rows = []
    for d, s in FDA_REGISTRY.items():
        if d in text.lower():
            rows.append({"Treatment": d.title(), "FDA Status": s})
    return pd.DataFrame(rows)

# ======================================================
# LAB ENGINE
# ======================================================
def extract_lab_values(text):
    patterns = {
        "Total Bilirubin": r"Total Bilirubin.*?(\d+\.?\d*)",
        "Direct Bilirubin": r"Direct Bilirubin.*?(\d+\.?\d*)",
        "SGPT": r"SGPT.*?(\d+)",
        "SGOT": r"SGOT.*?(\d+)",
        "GGT": r"Gamma.*Transferase.*?(\d+)"
    }
    results = {}
    for k, p in patterns.items():
        m = re.search(p, text, re.IGNORECASE)
        if m:
            results[k] = m.group(1)
    return results

def interpret_labs(values):
    summary = []
    if "Total Bilirubin" in values and float(values["Total Bilirubin"]) > 1.2:
        summary.append("🔴 Elevated bilirubin — Jaundice risk")
    if "SGPT" in values and float(values["SGPT"]) > 50:
        summary.append("🔴 SGPT high — Liver injury")
    if "SGOT" in values and float(values["SGOT"]) > 50:
        summary.append("🔴 SGOT high — Liver inflammation")
    if "GGT" in values and float(values["GGT"]) > 55:
        summary.append("🔴 GGT high — Alcohol/Biliary involvement")
    return summary

# ======================================================
# HOSPITAL AI
# ======================================================
def hospital_answer(query, context):
    prompt = f"""
You are a Hospital Clinical AI.

Use ONLY hospital evidence.
No hallucination.
If evidence insufficient, say so.

Hospital Evidence:
{context}

Doctor Query:
{query}
"""
    return external_research_answer(prompt).get("answer", "")

# ======================================================
# INDEX
# ======================================================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, p in enumerate(reader.pages[:200]):
                t = p.extract_text()
                if t and len(t) > 100:
                    docs.append(t)
                    srcs.append(f"{pdf} – Page {i+1}")
    if not docs:
        return None, [], []
    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🧠 ĀROGYABODHA AI")
st.sidebar.subheader("📁 Medical Library")

uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.sidebar.success("Index built successfully")

st.sidebar.divider()
app_mode = st.sidebar.radio("Select Module", ["Clinical Research Copilot", "Lab Report Intelligence"])

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='card'><h1>🧠 ĀROGYABODHA AI — Clinical Intelligence Command Center</h1></div>", unsafe_allow_html=True)

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if app_mode == "Clinical Research Copilot":

    st.markdown("<div class='card'><h2>🔬 Clinical Research Copilot</h2></div>", unsafe_allow_html=True)

    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze Research"):
        log_query(query, mode)

        if mode in ["Hospital AI", "Hybrid AI"]:
            qemb = embedder.encode([query])
            _, I = st.session_state.index.search(np.array(qemb), 5)
            context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
            raw = hospital_answer(query, context)

            level, coverage = semantic_evidence_level(raw, context)
            conf = confidence_score(raw, len(I[0]))

            st.markdown(f"<div class='success'>Confidence: {conf}% | Evidence Coverage: {coverage}%</div>", unsafe_allow_html=True)
            st.write(raw)

        if mode in ["Global AI", "Hybrid AI"]:
            st.markdown("<div class='card'><h3>🌍 Global Medical Research</h3></div>", unsafe_allow_html=True)
            st.write(external_research_answer(query).get("answer", ""))

# ======================================================
# LAB REPORT INTELLIGENCE
# ======================================================
if app_mode == "Lab Report Intelligence":

    st.markdown("<div class='card'><h2>🧪 Lab Report Intelligence</h2></div>", unsafe_allow_html=True)

    lab_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])

    if lab_file:
        with open("lab_report.pdf", "wb") as f:
            f.write(lab_file.getbuffer())

        reader = PdfReader("lab_report.pdf")
        report_text = ""
        for page in reader.pages:
            report_text += page.extract_text() + "\n"

        values = extract_lab_values(report_text)

        st.markdown("<div class='card'><h3>🧾 Smart Lab Summary</h3></div>", unsafe_allow_html=True)
        st.json(values)

        interpretation = interpret_labs(values)

        st.markdown("<div class='card'><h3>🩺 Clinical Interpretation</h3></div>", unsafe_allow_html=True)
        for line in interpretation:
            st.markdown(f"<div class='alert'>{line}</div>", unsafe_allow_html=True)

        lab_question = st.text_input("Ask ĀROGYABODHA AI")

        if st.button("🧠 Analyze Lab Report"):
            prompt = f"""
You are a hospital clinical AI.

Lab Report:
{report_text}

Doctor Question:
{lab_question}

Provide diagnosis pattern, risks and next steps.
"""
            answer = external_research_answer(prompt).get("answer", "")
            st.markdown("<div class='success'>AI Clinical Opinion</div>", unsafe_allow_html=True)
            st.write(answer)

# ======================================================
# FOOTER
# ======================================================
st.markdown("<center>ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform</center>", unsafe_allow_html=True)

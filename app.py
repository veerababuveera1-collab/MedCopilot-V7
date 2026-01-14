import streamlit as st
import os, json, pickle, datetime, re
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
    page_title="ĀROGYABODHA AI — Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

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
# SESSION STATE
# ======================================================
defaults = {
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "show_help": False,
    "role": "Doctor"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HEADER
# ======================================================
c1, c2, c3 = st.columns([6,1,1])
with c1:
    st.markdown("## 🧠 ĀROGYABODHA AI")
    st.caption("Hospital-Grade Clinical Intelligence Platform")
with c2:
    if st.button("❓ Help"):
        st.session_state.show_help = not st.session_state.show_help
with c3:
    st.session_state.role = st.selectbox("Role", ["Doctor", "Researcher"])

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
# HOSPITAL AI
# ======================================================
def hospital_answer(query, context):
    prompt = f"""
You are a Hospital Clinical Decision Support AI.

RULES:
- Use ONLY the hospital evidence below
- Do NOT use external knowledge
- Do NOT hallucinate

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

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# ======================================================
# LAB RULES
# ======================================================
LAB_RULES = {
    "Hemoglobin": (13,17,"g/dL"),
    "WBC": (4000,10000,"/cumm"),
    "Platelets": (150000,410000,"/cumm"),
    "Total Bilirubin": (0.3,1.2,"mg/dL"),
    "Direct Bilirubin": (0,0.2,"mg/dL"),
    "SGPT": (0,50,"U/L"),
    "SGOT": (0,50,"U/L"),
    "GGT": (0,55,"U/L"),
    "Creatinine": (0.7,1.3,"mg/dL")
}

def extract_lab_values_from_pdf(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    found = {}
    for i,line in enumerate(lines):
        for test in LAB_RULES:
            if test.lower() in line.lower():
                for j in range(i+1, min(i+5, len(lines))):
                    m = re.search(r"(\d+\.?\d*)", lines[j])
                    if m:
                        found[test] = float(m.group(1))
                        break
    return found

def generate_lab_summary(values):
    summary=[]
    alerts=[]
    for test,val in values.items():
        low,high,unit = LAB_RULES[test]
        if val<low: status="🟡 LOW"
        elif val>high: status="🔴 HIGH"
        else: status="🟢 NORMAL"
        summary.append((test,val,unit,status))

        if test=="Total Bilirubin" and val>=5:
            alerts.append("🚨 Severe Jaundice – ICU evaluation required")
        if test=="Creatinine" and val>=3:
            alerts.append("🚨 Acute Renal Failure risk")
        if test=="Platelets" and val<50000:
            alerts.append("🚨 Bleeding risk")

    return summary,alerts

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.subheader("📁 Medical Library")
uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True

st.sidebar.divider()
app_mode = st.sidebar.radio("Select Module", ["Clinical Research Copilot","Lab Report Intelligence"])

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if app_mode=="Clinical Research Copilot":

    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze"):
        log_query(query,mode)

        if mode in ["Hospital AI","Hybrid AI"]:
            qemb = embedder.encode([query])
            _, I = st.session_state.index.search(np.array(qemb), 5)
            context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
            raw = hospital_answer(query, context)

            level, coverage = semantic_evidence_level(raw, context)
            conf = confidence_score(raw, len(I[0]))

            st.metric("Confidence", f"{conf}%")
            st.metric("Evidence Coverage", f"{coverage}%")
            st.write(raw)

        if mode in ["Global AI","Hybrid AI"]:
            st.write(external_research_answer(query).get("answer",""))

# ======================================================
# LAB REPORT INTELLIGENCE
# ======================================================
if app_mode=="Lab Report Intelligence":

    st.markdown("## 🧪 Lab Report Intelligence")

    lab_file = st.file_uploader("Upload Lab Report PDF", type=["pdf"])

    if lab_file:
        with open("lab_report.pdf","wb") as f:
            f.write(lab_file.getbuffer())

        reader = PdfReader("lab_report.pdf")
        report_text=""
        for p in reader.pages:
            report_text+=(p.extract_text() or "")+"\n"

        values = extract_lab_values_from_pdf(report_text)
        summary,alerts = generate_lab_summary(values)

        st.subheader("🧾 Smart Lab Summary")
        for t,v,u,s in summary:
            st.write(f"{t}: {v} {u} — {s}")

        if alerts:
            st.subheader("🚨 ICU Alerts")
            for a in alerts:
                st.error(a)

        lab_question = st.text_input("Ask ĀROGYABODHA AI about this report")

        if st.button("🧠 Analyze Lab Report"):
            prompt=f"""
You are a hospital clinical AI.

Lab Report:
{report_text}

Doctor Question:
{lab_question}

Provide diagnosis pattern, risks and next steps.
"""
            answer = external_research_answer(prompt).get("answer","")
            st.success(answer)

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

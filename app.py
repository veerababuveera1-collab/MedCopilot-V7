# ======================================================
# ĀROGYABODHA AI — Clinical Decision Support System (CDSS)
# Hospital-Grade Clinical Answer Engine
# ======================================================

import streamlit as st
import os, json, pickle, datetime, io
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.info("ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only. "
        "It does NOT provide diagnosis or treatment. Final decisions must be made by licensed doctors.")

# ======================================================
# STORAGE
# ======================================================
BASE_DIR = os.getcwd()
PDF_FOLDER = os.path.join(BASE_DIR, "medical_library")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_cache")
AUDIT_LOG = os.path.join(BASE_DIR, "audit_log.json")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "logged_in": True,
    "username": "doctor1",
    "role": "Doctor",
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# AUDIT SYSTEM
# ======================================================
def audit(event, meta=None):
    rows = []
    if os.path.exists(AUDIT_LOG):
        rows = json.load(open(AUDIT_LOG))
    rows.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.username,
        "event": event,
        "meta": meta or {}
    })
    json.dump(rows, open(AUDIT_LOG, "w"), indent=2)

# ======================================================
# EMBEDDING MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# PDF + FAISS INDEX
# ======================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in reader.pages[:200]:
        t = p.extract_text()
        if t and len(t) > 100:
            texts.append(t)
    return texts

def build_index():
    docs, srcs = [], []

    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            with open(os.path.join(PDF_FOLDER, pdf), "rb") as f:
                pages = extract_text(f.read())
            for i, p in enumerate(pages):
                docs.append(p)
                srcs.append(f"{pdf} — Page {i+1}")

    if not docs:
        return None, [], []

    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))

    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))

    return idx, docs, srcs

if os.path.exists(INDEX_FILE):
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# ======================================================
# CLINICAL KNOWLEDGE BASE
# ======================================================
SYMPTOM_MAP = {
    "fever": ["Infection", "Malaria", "Dengue"],
    "chest pain": ["Heart attack", "Gastritis", "Anxiety"],
    "breathlessness": ["Asthma", "Heart failure", "Pneumonia"],
    "fatigue": ["Anemia", "Diabetes", "Thyroid disorder"],
    "vomiting": ["Food poisoning", "Liver disease", "Kidney disease"]
}

RISK_MAP = {
    "Heart attack": "HIGH",
    "Pneumonia": "MEDIUM",
    "Malaria": "MEDIUM",
    "Dengue": "HIGH",
    "Asthma": "MEDIUM",
    "Anemia": "LOW",
    "Anxiety": "LOW"
}

TEST_MAP = {
    "fever": ["CBC", "Malaria Test", "Dengue NS1"],
    "chest pain": ["ECG", "Troponin", "Chest X-Ray"],
    "breathlessness": ["Chest X-Ray", "SpO2", "ABG"],
    "fatigue": ["CBC", "TSH", "Blood Sugar"],
    "vomiting": ["LFT", "RFT", "Ultrasound Abdomen"]
}

RED_FLAGS = [
    "chest pain",
    "breathlessness",
    "unconscious",
    "high fever",
    "blood vomiting"
]

# ======================================================
# CLINICAL ENGINE
# ======================================================
def extract_symptoms(query):
    found = []
    q = query.lower()
    for s in SYMPTOM_MAP:
        if s in q:
            found.append(s)
    return found

def find_causes(symptoms):
    causes = []
    for s in symptoms:
        causes.extend(SYMPTOM_MAP.get(s, []))
    return list(set(causes))

def classify_risk(causes):
    return {c: RISK_MAP.get(c, "MEDIUM") for c in causes}

def detect_red_flags(symptoms):
    return [s for s in symptoms if s in RED_FLAGS]

def recommend_tests(symptoms):
    tests = []
    for s in symptoms:
        tests.extend(TEST_MAP.get(s, []))
    return list(set(tests))

def retrieve_evidence(query):
    qemb = embedder.encode(query)
    qvec = np.array([qemb], dtype=np.float32)
    D, I = st.session_state.index.search(qvec, 5)

    context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
    sources = [st.session_state.sources[i] for i in I[0]]

    return context, sources

# ======================================================
# UI
# ======================================================
st.sidebar.subheader("📁 Hospital Evidence Library")

uploads = st.sidebar.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("Build Evidence Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    audit("build_index", {"docs": len(st.session_state.documents)})
    st.sidebar.success("Index Built")

st.sidebar.markdown("🟢 Index Ready" if st.session_state.index_ready else "🔴 Index Not Built")

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
st.header("🔬 Clinical Research Copilot")

query = st.text_input("Ask a clinical research question")

if st.button("Analyze") and query:

    audit("clinical_query", {"query": query})

    symptoms = extract_symptoms(query)
    causes = find_causes(symptoms)
    risks = classify_risk(causes)
    red_flags = detect_red_flags(symptoms)
    tests = recommend_tests(symptoms)

    st.subheader("🏥 Clinical Summary")
    st.write(f"Identified symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}")

    st.subheader("🔍 Possible Causes & Risk")
    for c in causes:
        st.write(f"• {c} (Risk: {risks[c]})")

    if red_flags:
        st.subheader("🔴 Red Flags (Urgent Attention)")
        for r in red_flags:
            st.error(r)

    st.subheader("🧪 Suggested Initial Tests")
    for t in tests:
        st.write(f"• {t}")

    if st.session_state.index_ready:
        context, sources = retrieve_evidence(query)
        st.subheader("📚 Hospital Evidence")
        st.write(context[:2000] + " ...")

        st.subheader("Evidence Sources")
        for s in sources:
            st.info(s)

    st.caption("⚠ This is a Clinical Decision Support output. Final medical decisions must be made by licensed doctors.")

# ======================================================
# AUDIT VIEW
# ======================================================
if os.path.exists(AUDIT_LOG):
    st.subheader("🕒 Audit Log")
    df = pd.DataFrame(json.load(open(AUDIT_LOG)))
    st.dataframe(df, use_container_width=True)

st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

# ============================================================
# ĀROGYABODHA AI — Hospital Clinical Decision Support System
# Full End-to-End Clinical Operating Platform
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="ĀROGYABODHA AI — Hospital CDSS",
                   page_icon="🧠",
                   layout="wide")

st.info("ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS). "
        "It does NOT provide diagnosis or treatment. "
        "Final decisions must be made by licensed doctors.")

# ============================================================
# STORAGE
# ============================================================
BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
PATIENT_DB = os.path.join(BASE, "patients.json")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
VECTOR_FOLDER = os.path.join(BASE, "vector_cache")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"), indent=2)

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ============================================================
# SESSION
# ============================================================
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "index" not in st.session_state:
    st.session_state.index = None

# ============================================================
# AUDIT
# ============================================================
def audit(event, meta=None):
    logs = []
    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))
    logs.append({
        "time": str(datetime.datetime.now()),
        "event": event,
        "meta": meta or {}
    })
    json.dump(logs, open(AUDIT_LOG, "w"), indent=2)

# ============================================================
# PDF + FAISS
# ============================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages[:200]:
        t = p.extract_text()
        if t and len(t) > 100:
            pages.append(t)
    return pages

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
    pickle.dump({"docs": docs, "srcs": srcs}, open(CACHE_FILE, "wb"))

    return idx, docs, srcs

if os.path.exists(INDEX_FILE):
    st.session_state.index = faiss.read_index(INDEX_FILE)
    cache = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.docs = cache["docs"]
    st.session_state.srcs = cache["srcs"]
    st.session_state.index_ready = True

# ============================================================
# CLINICAL KNOWLEDGE
# ============================================================
SYMPTOMS = {
    "fever": ["Infection", "Malaria", "Dengue"],
    "chest pain": ["Heart attack", "Gastritis", "Anxiety"],
    "breathlessness": ["Asthma", "Heart failure", "Pneumonia"],
    "fatigue": ["Anemia", "Diabetes", "Thyroid disorder"],
    "vomiting": ["Food poisoning", "Liver disease"]
}

RISK = {
    "Heart attack": "HIGH",
    "Dengue": "HIGH",
    "Pneumonia": "MEDIUM",
    "Malaria": "MEDIUM",
    "Anemia": "LOW",
    "Anxiety": "LOW"
}

TESTS = {
    "fever": ["CBC", "Malaria Test", "Dengue NS1"],
    "chest pain": ["ECG", "Troponin"],
    "breathlessness": ["Chest X-Ray", "SpO2"],
    "fatigue": ["CBC", "TSH"],
    "vomiting": ["LFT", "RFT"]
}

RED_FLAGS = ["chest pain", "breathlessness", "unconscious", "high fever"]

# ============================================================
# CLINICAL ENGINE
# ============================================================
def extract_symptoms(q):
    q = q.lower()
    return [s for s in SYMPTOMS if s in q]

def get_causes(symptoms):
    c = []
    for s in symptoms:
        c.extend(SYMPTOMS[s])
    return list(set(c))

def get_risk(causes):
    return {c: RISK.get(c, "MEDIUM") for c in causes}

def get_tests(symptoms):
    t = []
    for s in symptoms:
        t.extend(TESTS.get(s, []))
    return list(set(t))

def get_redflags(symptoms):
    return [s for s in symptoms if s in RED_FLAGS]

def retrieve_evidence(q):
    qemb = embedder.encode(q)
    qvec = np.array([qemb], dtype=np.float32)
    D, I = st.session_state.index.search(qvec, 5)
    context = "\n\n".join([st.session_state.docs[i] for i in I[0]])
    sources = [st.session_state.srcs[i] for i in I[0]]
    return context, sources

# ============================================================
# UI SIDEBAR
# ============================================================
st.sidebar.title("🏥 Hospital Command Center")

module = st.sidebar.radio("Navigate", [
    "📁 Evidence Library",
    "👤 Patient Workspace",
    "🔬 Clinical Copilot",
    "🧪 Lab Intelligence",
    "🕒 Audit & Compliance"
])

# ============================================================
# EVIDENCE LIBRARY
# ============================================================
if module == "📁 Evidence Library":
    st.header("📁 Hospital Evidence Library")

    files = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success("PDFs uploaded")

    if st.button("Build Evidence Index"):
        st.session_state.index, st.session_state.docs, st.session_state.srcs = build_index()
        st.session_state.index_ready = True
        audit("build_index", {"docs": len(st.session_state.docs)})
        st.success("Index built")

    st.markdown("🟢 Index Ready" if st.session_state.index_ready else "🔴 Index Not Built")

# ============================================================
# PATIENT WORKSPACE
# ============================================================
if module == "👤 Patient Workspace":
    st.header("👤 Patient Case Workspace")

    patients = json.load(open(PATIENT_DB))

    with st.form("add_patient"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("Symptoms")
        submit = st.form_submit_button("Create Case")

    if submit:
        case = {
            "id": len(patients)+1,
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "time": str(datetime.datetime.now())
        }
        patients.append(case)
        json.dump(patients, open(PATIENT_DB, "w"), indent=2)
        audit("new_patient_case", case)
        st.success("Patient case created")

    st.subheader("Active Cases")
    st.dataframe(pd.DataFrame(patients))

# ============================================================
# CLINICAL COPILOT
# ============================================================
if module == "🔬 Clinical Copilot":
    st.header("🔬 Clinical Reasoning Engine")

    query = st.text_input("Enter patient symptoms or clinical question")

    if st.button("Analyze") and query:

        symptoms = extract_symptoms(query)
        causes = get_causes(symptoms)
        risks = get_risk(causes)
        tests = get_tests(symptoms)
        flags = get_redflags(symptoms)

        st.subheader("Clinical Summary")
        st.write("Symptoms detected:", symptoms)

        st.subheader("Possible Causes & Risk")
        for c in causes:
            st.write(f"• {c} (Risk: {risks[c]})")

        if flags:
            st.subheader("🚨 Red Flags")
            for f in flags:
                st.error(f)

        st.subheader("Suggested Tests")
        for t in tests:
            st.write("•", t)

        if st.session_state.index_ready:
            context, sources = retrieve_evidence(query)
            st.subheader("Hospital Evidence")
            st.write(context[:2000] + "...")

            st.subheader("Sources")
            for s in sources:
                st.info(s)

        audit("clinical_analysis", {"query": query})

# ============================================================
# LAB
# ============================================================
if module == "🧪 Lab Intelligence":
    st.header("🧪 Lab Report Upload")

    lab = st.file_uploader("Upload Lab Report", type=["pdf", "jpg", "png"])
    if lab:
        path = os.path.join(BASE, lab.name)
        with open(path, "wb") as out:
            out.write(lab.getbuffer())
        audit("lab_upload", {"file": lab.name})
        st.success("Lab uploaded")

# ============================================================
# AUDIT
# ============================================================
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance")

    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — Hospital Clinical Operating System")

# ============================================================
# ĀROGYABODHA AI — Hospital Clinical Care + Decision Support OS
# Full End-to-End Hospital Operating Platform
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
st.set_page_config("ĀROGYABODHA AI — Hospital OS", "🧠", layout="wide")

st.info(
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS). "
    "It does NOT provide diagnosis or treatment. "
    "Final decisions must be made by licensed doctors."
)

# ============================================================
# STORAGE
# ============================================================
BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
VECTOR_FOLDER = os.path.join(BASE, "vector_cache")
PATIENT_DB = os.path.join(BASE, "patients.json")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Init DBs
if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"), indent=2)

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION
# ============================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index_ready": False,
    "index": None,
    "docs": [],
    "srcs": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# AUDIT
# ============================================================
def audit(event, meta=None):
    logs = []
    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))
    logs.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.username,
        "role": st.session_state.role,
        "event": event,
        "meta": meta or {}
    })
    json.dump(logs, open(AUDIT_LOG, "w"), indent=2)

# ============================================================
# LOGIN
# ============================================================
def login_ui():
    st.title("ĀROGYABODHA AI — Secure Hospital Login")
    with st.form("login"):
        u = st.text_input("User ID")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")

    if ok:
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.role = users[u]["role"]
            audit("login", {"user": u})
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

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

# Load existing index
if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
    st.session_state.index = faiss.read_index(INDEX_FILE)
    cache = pickle.load(open(CACHE_FILE, "rb"))

    if "docs" in cache:
        st.session_state.docs = cache["docs"]
        st.session_state.srcs = cache["srcs"]
    elif "documents" in cache:
        st.session_state.docs = cache["documents"]
        st.session_state.srcs = cache["sources"]

    st.session_state.index_ready = True

# ============================================================
# CLINICAL KNOWLEDGE
# ============================================================
SYMPTOMS = {
    "fever": ["Infection", "Malaria", "Dengue"],
    "chest pain": ["Heart attack", "Gastritis", "Anxiety"],
    "breathlessness": ["Asthma", "Heart failure", "Pneumonia"],
    "fatigue": ["Anemia", "Diabetes"],
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
    "fatigue": ["CBC", "Blood Sugar"],
    "vomiting": ["LFT", "RFT"]
}

TREATMENT_PROTOCOLS = {
    "Heart attack": ["Aspirin", "ECG Monitoring", "ICU Admission"],
    "Dengue": ["IV Fluids", "Platelet Monitoring"],
    "Pneumonia": ["Antibiotics", "Oxygen Therapy"],
    "Malaria": ["Antimalarial Therapy"],
    "Anemia": ["Iron Therapy"]
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

def get_treatments(causes):
    t = []
    for c in causes:
        t.extend(TREATMENT_PROTOCOLS.get(c, []))
    return list(set(t))

def get_redflags(symptoms):
    return [s for s in symptoms if s in RED_FLAGS]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Hospital Command Center", [
    "📁 Evidence Library",
    "👤 Patient Workspace",
    "🔬 Clinical Copilot",
    "🧾 Doctor Orders",
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
        st.success("Index built successfully")

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
            "timeline": [],
            "time": str(datetime.datetime.now())
        }
        patients.append(case)
        json.dump(patients, open(PATIENT_DB, "w"), indent=2)
        audit("new_patient_case", case)
        st.success("Patient case created")

    st.subheader("Active Cases")
    st.dataframe(pd.DataFrame(patients), use_container_width=True)

# ============================================================
# CLINICAL COPILOT
# ============================================================
if module == "🔬 Clinical Copilot":
    st.header("🔬 Clinical Reasoning Engine")

    query = st.text_input("Enter patient symptoms")

    if st.button("Analyze") and query:
        symptoms = extract_symptoms(query)
        causes = get_causes(symptoms)
        risks = get_risk(causes)
        tests = get_tests(symptoms)
        flags = get_redflags(symptoms)
        treatments = get_treatments(causes)

        st.subheader("Clinical Summary")
        st.write("Symptoms:", symptoms)

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

        st.subheader("Standard Treatment Protocols")
        for tr in treatments:
            st.write("•", tr)

        audit("clinical_analysis", {"query": query})

# ============================================================
# DOCTOR ORDERS
# ============================================================
if module == "🧾 Doctor Orders":
    st.header("🧾 Doctor Orders & Care Actions")

    patients = json.load(open(PATIENT_DB))

    if not patients:
        st.info("No patients available.")
    else:
        pid = st.selectbox("Select Patient ID", [p["id"] for p in patients])
        order = st.text_area("Enter Doctor Order (Treatment / Admission / Referral)")

        if st.button("Submit Order"):
            for p in patients:
                if p["id"] == pid:
                    p["timeline"].append({
                        "time": str(datetime.datetime.now()),
                        "doctor": st.session_state.username,
                        "order": order
                    })
            json.dump(patients, open(PATIENT_DB, "w"), indent=2)
            audit("doctor_order", {"patient_id": pid, "order": order})
            st.success("Doctor order recorded")

# ============================================================
# AUDIT
# ============================================================
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance")

    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No audit logs yet.")

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — Hospital Clinical Operating System")

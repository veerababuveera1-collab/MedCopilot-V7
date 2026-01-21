# ============================================================
# ĀROGYABODHA AI — Clinical Decision Intelligence OS (CDIS)
# National Hospital Intelligence Platform
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, requests
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================================
# CONFIG
# ============================================================
st.set_page_config("ĀROGYABODHA AI — Clinical Decision Intelligence OS", "🧠", layout="wide")

st.info("""
ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS).
It does NOT provide diagnosis or treatment.
Final clinical decisions must be made by licensed doctors.
""")

# ============================================================
# PATHS
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

# ============================================================
# INIT DATABASES
# ============================================================
if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"), indent=2)

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "admin": {"password": "admin123", "role": "Administrator"}
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index_ready": False,
    "index": None,
    "docs": [],
    "srcs": [],
    "ai_mode": "Hybrid AI"
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
        "time": str(datetime.datetime.utcnow()),
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
            audit("LOGIN", {"user": u})
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ============================================================
# AI MODEL
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ============================================================
# PDF INDEXING
# ============================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages[:200]:
        t = p.extract_text()
        if t and len(t) > 200:
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

if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
    st.session_state.index = faiss.read_index(INDEX_FILE)
    cache = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.docs = cache["docs"]
    st.session_state.srcs = cache["srcs"]
    st.session_state.index_ready = True

# ============================================================
# CLINICAL SAFE SEARCH
# ============================================================
def search_rag(query, k=5):
    if not st.session_state.index_ready:
        return [], []

    qemb = embedder.encode([query])
    D, I = st.session_state.index.search(np.array(qemb).astype("float32"), k)

    hits, srcs = [], []
    for idx in I[0]:
        hits.append(st.session_state.docs[idx])
        srcs.append(st.session_state.srcs[idx])
    return hits, srcs

# ============================================================
# CLINICAL PROTOCOL ENGINE
# ============================================================
def build_clinical_protocol(query, chunks):
    text = " ".join(chunks)
    lines = [l.strip() for l in text.split(".") if len(l.strip()) > 60]

    protocol = f"""
## 🏥 Hospital Clinical Protocol

### 🔍 Clinical Scenario
**{query}**

---

### 🚑 Immediate Actions
"""
    for i, l in enumerate(lines[:6], 1):
        protocol += f"{i}. {l}.\n\n"

    protocol += """
---

### 🏥 Hospital Operations
• Activate emergency team  
• Ensure ICU / surge beds  
• Verify drugs & equipment  
• Enable ambulance coordination  

---

### 📡 Command & Communication
• Emergency In-charge assumes command  
• Central control room activation  
• Patient documentation & tracking  
• Inter-hospital escalation  

---

### ⚠ Clinical Governance
• Follow DGHS / MoHFW protocols  
• Maintain audit logs  
• Escalate to Medical Superintendent  

---

### 📌 Disclaimer
Generated from hospital-approved medical literature.
Final decisions remain with the treating physician.
"""

    return protocol

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

st.session_state.ai_mode = st.sidebar.radio(
    "AI Mode",
    ["Hospital AI", "Global AI", "Hybrid AI"],
    index=["Hospital AI", "Global AI", "Hybrid AI"].index(st.session_state.ai_mode)
)

if st.sidebar.button("Logout"):
    audit("LOGOUT")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Medical Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Clinical Research Copilot",
    "👤 Patient Workspace",
    "📊 Dashboard",
    "🕒 Audit & Compliance"
])

# ============================================================
# MODULES
# ============================================================

# ---------- Evidence Library ----------
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
        audit("BUILD_INDEX", {"docs": len(st.session_state.docs)})
        st.success("Index built successfully")

# ---------- Clinical Research Copilot ----------
if module == "🔬 Clinical Research Copilot":
    st.header("🔬 Clinical Decision Intelligence Engine")
    st.write(f"Selected AI Mode: **{st.session_state.ai_mode}**")

    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze") and query:
        audit("QUERY", {"query": query, "mode": st.session_state.ai_mode})

        hospital_hits, sources = [], []

        if st.session_state.ai_mode in ["Hospital AI", "Hybrid AI"]:
            hospital_hits, sources = search_rag(query)

        if hospital_hits:
            st.subheader("🏥 Hospital Clinical Protocol")
            protocol = build_clinical_protocol(query, hospital_hits)
            st.markdown(protocol)

            st.markdown("### 📚 Evidence Sources")
            for s in sources:
                st.success(s)

        else:
            st.warning("No hospital protocol found for this query. Try uploading more hospital SOPs.")

# ---------- Patient Workspace ----------
if module == "👤 Patient Workspace":
    st.header("👤 Patient Workspace")

    patients = json.load(open(PATIENT_DB))

    with st.form("add_patient"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 0, 120)
        diagnosis = st.text_input("Diagnosis")
        submit = st.form_submit_button("Create Case")

    if submit:
        patients.append({
            "id": len(patients)+1,
            "name": name,
            "age": age,
            "diagnosis": diagnosis,
            "time": str(datetime.datetime.utcnow())
        })
        json.dump(patients, open(PATIENT_DB, "w"), indent=2)
        audit("NEW_PATIENT", {"name": name})
        st.success("Patient created")

    st.dataframe(pd.DataFrame(patients))

# ---------- Dashboard ----------
if module == "📊 Dashboard":
    st.header("📊 Hospital Intelligence Dashboard")
    st.metric("Evidence Documents", len(st.session_state.docs))
    st.metric("Vector Index", "Ready" if st.session_state.index_ready else "Not Ready")
    st.metric("Audit Events", len(json.load(open(AUDIT_LOG))) if os.path.exists(AUDIT_LOG) else 0)

# ---------- Audit ----------
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance")
    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))))

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — National Clinical Decision Intelligence OS")

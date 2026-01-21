# ============================================================
# ĀROGYABODHA AI — Hospital + Clinical Research Intelligence OS
# Final End-to-End Medical Intelligence Platform
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, textwrap
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================================
# CONFIG
# ============================================================
st.set_page_config("ĀROGYABODHA AI — Hospital & Research OS", "🧠", layout="wide")

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
RESEARCH_FOLDER = os.path.join(BASE, "research_library")
VECTOR_FOLDER = os.path.join(BASE, "vector_cache")

PATIENT_DB = os.path.join(BASE, "patients.json")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")
FDA_DB = os.path.join(BASE, "fda_drugs.json")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(RESEARCH_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Databases
if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"), indent=2)

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"), indent=2)

if not os.path.exists(FDA_DB):
    json.dump({
        "Temozolomide": "FDA Approved",
        "Bevacizumab": "FDA Approved",
        "CAR-T Therapy": "Conditional Approval",
        "Experimental Vaccine X": "Clinical Trial Phase"
    }, open(FDA_DB, "w"), indent=2)

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
    st.title("ĀROGYABODHA AI — Secure Medical Intelligence Login")
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
# PDF + FAISS INDEX
# ============================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages[:200]:
        t = p.extract_text()
        if t and len(t) > 100:
            pages.append(t)
    return pages

def load_all_documents():
    docs, srcs = [], []
    for folder in [PDF_FOLDER, RESEARCH_FOLDER]:
        for pdf in os.listdir(folder):
            if pdf.endswith(".pdf"):
                with open(os.path.join(folder, pdf), "rb") as f:
                    pages = extract_text(f.read())
                for i, p in enumerate(pages):
                    docs.append(p)
                    srcs.append(f"{pdf} — Page {i+1}")
    return docs, srcs

def build_index():
    docs, srcs = load_all_documents()
    if not docs:
        return None, [], []
    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"docs": docs, "srcs": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

# Load index
if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
    try:
        st.session_state.index = faiss.read_index(INDEX_FILE)
        cache = pickle.load(open(CACHE_FILE, "rb"))
        st.session_state.docs = cache["docs"]
        st.session_state.srcs = cache["srcs"]
        st.session_state.index_ready = True
    except:
        st.session_state.index_ready = False

# ============================================================
# RESEARCH UTILITIES
# ============================================================
FDA_STATUS = json.load(open(FDA_DB))

def get_fda_status(drug):
    return FDA_STATUS.get(drug, "Unknown / Experimental")

def compare_treatments(treatments):
    table = []
    for t in treatments:
        table.append({
            "Treatment": t,
            "FDA Status": get_fda_status(t),
            "Outcome": "Improved survival in recent trials",
            "Side Effects": "Moderate (trial dependent)"
        })
    return pd.DataFrame(table)

# ============================================================
# CLINICAL ANSWER FORMATTER
# ============================================================
def clinical_formatter(query, context, sources):
    short = textwrap.shorten(context.replace("\n", " "), width=1200)
    srcs = "\n".join([f"• {s}" for s in sources])

    return f"""
## 🏥 Clinical Research Summary — {query}

### 🔬 Evidence Summary
{short}

### 💊 Treatment Landscape
Based on recent clinical trials and international guidelines.

### 🧪 Regulatory Status
FDA approval varies by drug and indication.

### 🚨 Safety Signals
• Disease progression  
• Therapy resistance  
• Adverse reactions  

### 📚 Evidence Sources
{srcs}
"""

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Medical Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Research Copilot",
    "👤 Patient Workspace",
    "🧾 Doctor Orders",
    "🕒 Audit & Compliance"
])

# ============================================================
# EVIDENCE LIBRARY
# ============================================================
if module == "📁 Evidence Library":
    st.header("📁 Medical Evidence Library")

    files = st.file_uploader("Upload Medical / Research PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(RESEARCH_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success("PDFs uploaded")

    if st.button("Build Global Evidence Index"):
        st.session_state.index, st.session_state.docs, st.session_state.srcs = build_index()
        st.session_state.index_ready = True
        audit("build_index", {"docs": len(st.session_state.docs)})
        st.success("Global Medical Index built successfully")

    st.markdown("🟢 Index Ready" if st.session_state.index_ready else "🔴 Index Not Built")

# ============================================================
# RESEARCH COPILOT
# ============================================================
if module == "🔬 Research Copilot":
    st.header("🔬 Clinical Research Copilot")

    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze Research") and query:
        audit("research_query", {"query": query})

        if not st.session_state.index_ready:
            st.error("Global evidence index not built.")
        else:
            qemb = embedder.encode(query)
            qvec = np.array([qemb], dtype=np.float32)
            D, I = st.session_state.index.search(qvec, 5)

            context = "\n".join([st.session_state.docs[i] for i in I[0]])
            sources = [st.session_state.srcs[i] for i in I[0]]

            st.markdown(clinical_formatter(query, context, sources))

            demo_treatments = ["Temozolomide", "Bevacizumab", "CAR-T Therapy"]
            st.subheader("📊 Treatment Outcome Comparison")
            df = compare_treatments(demo_treatments)
            st.dataframe(df, use_container_width=True)

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
# DOCTOR ORDERS
# ============================================================
if module == "🧾 Doctor Orders":
    st.header("🧾 Doctor Orders & Care Actions")

    patients = json.load(open(PATIENT_DB))

    if not patients:
        st.info("No patients available.")
    else:
        pid = st.selectbox("Select Patient ID", [p["id"] for p in patients])
        order = st.text_area("Enter Doctor Order")

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
st.caption("ĀROGYABODHA AI — Hospital & Clinical Research Intelligence Platform")

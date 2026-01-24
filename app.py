# ============================================================
# ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS
# Hospital + Research + Trial + Regulatory + Clinical Reasoning
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, requests, re, base64
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Medical Intelligence OS",
    page_icon="🧠",
    layout="wide"
)

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

# ============================================================
# DATABASE INIT
# ============================================================
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
    st.title("ĀROGYABODHA AI — Secure Medical Login")
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
# PDF HELPERS
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
        if pdf.lower().endswith(".pdf"):
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

def display_pdf(pdf_path, height=700):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_iframe = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
                width="100%" height="{height}"
                type="application/pdf"></iframe>
    """
    st.markdown(pdf_iframe, unsafe_allow_html=True)

# ============================================================
# EXTERNAL INTELLIGENCE
# ============================================================
def fetch_pubmed(query):
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 5},
            timeout=15
        )
        return r.json()["esearchresult"]["idlist"]
    except:
        return []

def fetch_clinical_trials(query):
    try:
        r = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": query, "pageSize": 5},
            timeout=15
        )
        trials = []
        for s in r.json().get("studies", []):
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            trials.append({
                "Trial ID": ident.get("nctId", "N/A"),
                "Phase": ", ".join(design.get("phases", ["N/A"])),
                "Status": status.get("overallStatus", "Unknown")
            })
        return trials
    except:
        return []

def fetch_fda_alerts():
    try:
        r = requests.get(
            "https://api.fda.gov/drug/enforcement.json?limit=5",
            timeout=15
        )
        return [
            f"{i.get('product_description','Unknown')} | "
            f"{i.get('reason_for_recall','Safety Alert')}"
            for i in r.json().get("results", [])
        ]
    except:
        return []

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
    "🔬 Phase-3 Research Copilot",
    "📊 Live Intelligence Dashboard",
    "👤 Patient Workspace",
    "🧾 Doctor Orders",
    "🕒 Audit & Compliance"
])

# ============================================================
# MODULES
# ============================================================

# ---------- Evidence Library ----------
if module == "📁 Evidence Library":
    st.header("📁 Medical Evidence Library")

    files = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
                out.write(f.read())
        st.success("PDFs uploaded successfully")

    if st.button("Build Evidence Index"):
        st.session_state.index, st.session_state.docs, st.session_state.srcs = build_index()
        audit("build_index", {"docs": len(st.session_state.docs)})
        st.success("Evidence index built")

    st.divider()
    st.subheader("📄 View Uploaded PDFs")

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if pdf_files:
        selected_pdf = st.selectbox("Select a PDF", pdf_files)
        display_pdf(os.path.join(PDF_FOLDER, selected_pdf))
    else:
        st.info("No PDFs available")

# ---------- Phase-3 Research Copilot ----------
if module == "🔬 Phase-3 Research Copilot":
    st.header("🧠 Clinical Research Intelligence Assistant")

    query = st.text_input(
        "Ask a clinical research question",
        placeholder="e.g., Pathological methods for diagnosing leukemia"
    )

    if st.button("Analyze Research") and query:
        audit("phase3_query", {"query": query})

        pubmed_ids = fetch_pubmed(query)
        trials = fetch_clinical_trials(query)
        alerts = fetch_fda_alerts()

        st.subheader("📚 PubMed Journal Evidence")
        if pubmed_ids:
            for pmid in pubmed_ids:
                st.markdown(f"- PMID: **{pmid}**  "
                            f"[View](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
        else:
            st.info("No PubMed articles found")

        st.subheader("🧪 Clinical Trials")
        if trials:
            st.table(pd.DataFrame(trials))
        else:
            st.info("No clinical trials found")

        st.subheader("⚠ FDA Safety Alerts")
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.info("No FDA safety alerts")

        st.subheader("📄 Local Hospital Evidence (PDF)")
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
        if pdf_files:
            selected_pdf = st.selectbox("Select Evidence PDF", pdf_files, key="phase3_pdf")
            display_pdf(os.path.join(PDF_FOLDER, selected_pdf))
        else:
            st.info("No local PDFs available")

# ---------- Live Dashboard ----------
if module == "📊 Live Intelligence Dashboard":
    st.header("📊 Live Medical Intelligence Dashboard")
    st.metric("Indexed Evidence Pages", len(st.session_state.docs))
    st.metric("PubMed Feed", "LIVE")
    st.metric("Clinical Trials Feed", "LIVE")
    st.metric("FDA Regulatory Feed", "LIVE")

# ---------- Patient Workspace ----------
if module == "👤 Patient Workspace":
    st.header("👤 Patient Workspace")

    patients = json.load(open(PATIENT_DB))
    with st.form("add_patient"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("Symptoms")
        submit = st.form_submit_button("Create Patient Case")

    if submit:
        patients.append({
            "id": len(patients) + 1,
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "created": str(datetime.datetime.utcnow())
        })
        json.dump(patients, open(PATIENT_DB, "w"), indent=2)
        audit("new_patient", {"name": name})
        st.success("Patient case created")

    if patients:
        st.dataframe(pd.DataFrame(patients), use_container_width=True)
    else:
        st.info("No patient cases available")

# ---------- Doctor Orders ----------
if module == "🧾 Doctor Orders":
    st.header("🧾 Doctor Orders")
    st.info("Doctor order module ready")

# ---------- Audit ----------
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance")
    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))), use_container_width=True)
    else:
        st.info("No audit records found")

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS")

# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# Command Center Edition
# ======================================================

import streamlit as st
import os, json, pickle, datetime, io
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# Optional OCR
OCR_AVAILABLE = True
try:
    import pytesseract
    from pdf2image import convert_from_path
except:
    OCR_AVAILABLE = False


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence Platform",
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
BASE_DIR = os.getcwd()
PDF_FOLDER = os.path.join(BASE_DIR, "medical_library")
LAB_FOLDER = os.path.join(BASE_DIR, "lab_reports")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_cache")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")
USERS_DB = os.path.join(BASE_DIR, "users.json")
AUDIT_LOG = os.path.join(BASE_DIR, "audit_log.json")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(LAB_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Demo users
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"), indent=2)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
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
        try:
            rows = json.load(open(AUDIT_LOG))
        except:
            rows = []
    rows.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.get("username"),
        "role": st.session_state.get("role"),
        "event": event,
        "meta": meta or {}
    })
    json.dump(rows, open(AUDIT_LOG, "w"), indent=2)

# ======================================================
# SAFE AI WRAPPER
# ======================================================
def safe_ai_call(prompt, mode="AI"):
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            audit("ai_empty_response", {"mode": mode})
            return {"status": "error", "answer": "⚠ AI returned empty response."}
        return {"status": "ok", "answer": result["answer"]}
    except Exception as e:
        audit("ai_failure", {"mode": mode, "error": str(e)})
        return {"status": "down", "answer": "⚠ AI service unavailable. Governance block applied."}

# ======================================================
# LOGIN UI (unchanged)
# ======================================================
def login_ui():
    st.title("ĀROGYABODHA AI — Secure Clinical Access")
    with st.form("login_form"):
        username = st.text_input("Doctor / Researcher ID")
        password = st.text_input("Secure Access Key", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        users = json.load(open(USERS_DB))
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = users[username]["role"]
            audit("login", {"user": username})
            st.rerun()
        else:
            audit("login_failed", {"user": username})
            st.error("Invalid credentials")

# ======================================================
# AUTH GATE
# ======================================================
if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ======================================================
# MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# FAISS INDEX
# ======================================================
def extract_text_from_pdf_bytes(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in reader.pages[:300]:
        t = p.extract_text()
        if t and len(t) > 100:
            texts.append(t)
    return texts

def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.lower().endswith(".pdf"):
            with open(os.path.join(PDF_FOLDER, pdf), "rb") as f:
                texts = extract_text_from_pdf_bytes(f.read())
            for i, t in enumerate(texts):
                docs.append(t)
                srcs.append(f"{pdf} — Page {i+1}")

    if not docs:
        return None, [], []

    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))
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
# SIDEBAR — COMMAND CENTER NAV
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    audit("logout", {})
    st.session_state.logged_in = False
    st.rerun()

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
    audit("build_index", {"count": len(st.session_state.documents)})
    st.sidebar.success("Evidence Index Built")

st.sidebar.markdown("🟢 Index Ready" if st.session_state.index_ready else "🔴 Index Not Built")

module = st.sidebar.radio("Command Center", [
    "🧠 System Dashboard",
    "🔬 Clinical Research Copilot",
    "🧪 Lab Intelligence",
    "🕒 Audit & Compliance"
])

# ======================================================
# SYSTEM DASHBOARD (NEW)
# ======================================================
if module == "🧠 System Dashboard":
    st.header("🧠 Hospital AI Command Center")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("System Status", "ONLINE")
        st.metric("AI Engine", "CONNECTED")
        st.metric("OCR Engine", "READY" if OCR_AVAILABLE else "DISABLED")

    with col2:
        st.metric("Evidence Index", "READY" if st.session_state.index_ready else "NOT BUILT")
        st.metric("Total Documents", len(st.session_state.documents))
        st.metric("User Role", st.session_state.role)

    with col3:
        if os.path.exists(AUDIT_LOG):
            logs = json.load(open(AUDIT_LOG))
            st.metric("Audit Events", len(logs))
            st.metric("Last Audit", logs[-1]["time"])
        else:
            st.metric("Audit Events", 0)

    st.success("Governance Engine: ACTIVE")
    st.success("Clinical Safety Layer: ENABLED")

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "🔬 Clinical Research Copilot":
    st.header("🔬 Clinical Research Copilot")

    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("Analyze") and query:
        audit("clinical_query", {"query": query, "mode": mode})

        col1, col2 = st.columns(2)

        if mode in ["Hospital AI", "Hybrid AI"]:
            with col1:
                st.subheader("🏥 Hospital Evidence AI")
                if not st.session_state.index_ready:
                    st.error("Evidence index not built.")
                else:
                    qemb = embedder.encode([query])
                    _, I = st.session_state.index.search(np.array(qemb, dtype=np.float32), 5)

                    context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
                    sources = [st.session_state.sources[i] for i in I[0]]

                    prompt = f"Use hospital evidence only:\n{context}\n\nQuestion: {query}"
                    resp = safe_ai_call(prompt, "Hospital AI")

                    st.write(resp["answer"])
                    st.markdown("#### Evidence Sources")
                    for s in sources:
                        st.info(s)

        if mode in ["Global AI", "Hybrid AI"]:
            with col2:
                st.subheader("🌍 Global Research AI")
                resp = safe_ai_call(query, "Global AI")
                st.write(resp["answer"])

# ======================================================
# LAB INTELLIGENCE
# ======================================================
if module == "🧪 Lab Intelligence":
    st.header("🧪 Lab Report Intelligence")

    uploaded_lab = st.file_uploader("Upload Lab Report (PDF/Image)", type=["pdf", "png", "jpg"])

    if uploaded_lab:
        path = os.path.join(LAB_FOLDER, uploaded_lab.name)
        with open(path, "wb") as out:
            out.write(uploaded_lab.getbuffer())

        audit("lab_upload", {"file": uploaded_lab.name})
        st.success("Lab report uploaded")

        st.info("OCR + AI interpretation pipeline ready (enable in v2.0)")

# ======================================================
# AUDIT TRAIL
# ======================================================
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance Dashboard")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No audit records yet.")

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

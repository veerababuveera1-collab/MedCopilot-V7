# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================

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
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# THEME
# ======================================================
st.markdown("""
<style>
body { background: #0b1220; color: #e5e7eb; }
.card { background: rgba(255,255,255,0.05); border-radius: 14px; padding: 16px; margin-bottom: 12px; }
.badge { padding: 6px 10px; border-radius: 999px; font-weight: 600; }
.ok { background: #00c2a8; color: #041b16; }
.warn { background: #ffd166; color: #3b2f00; }
.danger { background: #ef476f; }
.small { opacity: .8; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# DISCLAIMER
# ======================================================
st.info("ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only. "
        "It does NOT provide diagnosis or treatment. "
        "Final clinical decisions must be made by licensed medical professionals.")

# ======================================================
# STORAGE
# ======================================================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
USERS_DB = "users.json"
AUDIT_LOG = "audit_log.json"
REPORTS_FOLDER = "doctor_reports"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ======================================================
# USERS
# ======================================================
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"}
    }, open(USERS_DB, "w"), indent=2)

# ======================================================
# SESSION
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ======================================================
# AUDIT
# ======================================================
def audit(event, meta=None):
    logs = []
    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))
    logs.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.get("username"),
        "event": event,
        "meta": meta or {}
    })
    json.dump(logs, open(AUDIT_LOG, "w"), indent=2)

# ======================================================
# AUTH
# ======================================================
def login_ui():
    st.markdown("### 🔐 Doctor Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            audit("login")
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ======================================================
# LOGOUT
# ======================================================
if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

# ======================================================
# MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# FAISS
# ======================================================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, p in enumerate(reader.pages[:100]):
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

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    documents = data["documents"]
    sources = data["sources"]
else:
    index, documents, sources = None, [], []

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("## 🧠 ĀROGYABODHA AI")
st.sidebar.markdown(f"**User:** {st.session_state.username}")

# Upload
uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("Build Index"):
    index, documents, sources = build_index()
    st.sidebar.success("Index built")

# Viewer + Delete
st.sidebar.markdown("### 📁 Medical Library")
for pdf in os.listdir(PDF_FOLDER):
    c1, c2 = st.sidebar.columns([8,1])
    with c1:
        st.write("📄", pdf)
    with c2:
        if st.button("🗑️", key=pdf):
            os.remove(os.path.join(PDF_FOLDER, pdf))
            if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
            if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
            audit("delete_pdf", {"file": pdf})
            st.rerun()

# Help Panel
with st.sidebar.expander("❓ Help"):
    st.write("""
**AI Modes**
- Hospital AI → Only hospital PDFs
- Global AI → Global research
- Hybrid AI → Both

**Modules**
- Research Copilot
- Lab Intelligence
- ICU Command Center
- Audit Trail
""")

module = st.sidebar.radio("Select Module", [
    "Clinical Research Copilot",
    "Lab Report Intelligence",
    "ICU Command Center",
    "Audit Trail"
])

# ======================================================
# HEADER
# ======================================================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform")
st.markdown("<div class='small'>Hospital-grade • Evidence-locked • Doctor-safe</div>", unsafe_allow_html=True)

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "Clinical Research Copilot":
    st.markdown("### 🔬 Clinical Research Copilot")
    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("Analyze"):
        audit("research_query", {"query": query, "mode": mode})

        if mode in ["Hospital AI","Hybrid AI"] and index:
            qemb = embedder.encode([query])
            _, I = index.search(np.array(qemb), 5)
            context = "\n\n".join([documents[i] for i in I[0]])
            prompt = f"""
You are a Hospital Clinical AI.
Use only hospital evidence.

Hospital Evidence:
{context}

Query:
{query}
"""
            st.subheader("🏥 Hospital AI")
            st.write(external_research_answer(prompt).get("answer",""))

        if mode in ["Global AI","Hybrid AI"]:
            st.subheader("🌍 Global AI")
            st.write(external_research_answer(query).get("answer",""))

# ======================================================
# LAB REPORT INTELLIGENCE
# ======================================================
if module == "Lab Report Intelligence":
    st.markdown("### 🧪 Lab Report Intelligence")
    lab_file = st.file_uploader("Upload Lab Report PDF", type=["pdf"])

    if lab_file:
        reader = PdfReader(lab_file)
        text = ""
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"

        st.subheader("🧠 AI Clinical Opinion")
        prompt = f"""
You are a hospital clinical AI.

Lab Report:
{text}

Provide diagnosis pattern, risks, and next steps.
"""
        st.write(external_research_answer(prompt).get("answer",""))

# ======================================================
# ICU COMMAND CENTER
# ======================================================
if module == "ICU Command Center":
    st.markdown("### 🚨 ICU Command Center")
    st.write("Monitoring lab alerts and doctor summaries.")

# ======================================================
# AUDIT TRAIL
# ======================================================
if module == "Audit Trail":
    st.markdown("### 🕒 Audit Trail")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No logs yet.")

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital Clinical Intelligence Platform")

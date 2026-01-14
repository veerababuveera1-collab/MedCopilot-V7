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
# SAFE AI WRAPPER (NO CRASH)
# ======================================================
def safe_ai_call(prompt):
    try:
        result = external_research_answer(prompt)
        return result.get("answer", "⚠ AI returned empty response.")
    except Exception as e:
        return "⚠ AI service temporarily unavailable. Please try again later."

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
.card { background:#111827;padding:20px;border-radius:12px;margin-bottom:16px;border:1px solid #1f2937;}
.header { font-size:28px;font-weight:700;color:#e5e7eb }
.badge-ok {color:#10b981}
.badge-warn {color:#f59e0b}
.badge-danger {color:#ef4444}
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

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

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
# AUTH
# ======================================================
def login_ui():
    st.markdown("## 🔐 Doctor Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.role = users[u]["role"]
            st.success("Login successful")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='header'>🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform</div>", unsafe_allow_html=True)
st.caption("Hospital-grade • Evidence-locked • Doctor-safe")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown(f"### 👨‍⚕️ User: {st.session_state.username}")
st.sidebar.divider()

# Upload PDFs
st.sidebar.subheader("📁 Medical Library")
uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("Uploaded")

# List PDFs with delete
for pdf in os.listdir(PDF_FOLDER):
    if pdf.endswith(".pdf"):
        c1, c2 = st.sidebar.columns([4,1])
        with c1:
            st.write("📄", pdf)
        with c2:
            if st.button("🗑", key=pdf):
                os.remove(os.path.join(PDF_FOLDER, pdf))
                if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
                if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
                st.experimental_rerun()

# Help
with st.sidebar.expander("❓ Help"):
    st.write("""
**AI Modes**
- Hospital AI → Uses only hospital PDFs
- Global AI → Uses global research
- Hybrid AI → Compares both

**Safety**
- No diagnosis
- No prescription
- Evidence locked
""")

# Module selector
module = st.sidebar.radio("Select Module", [
    "Clinical Research Copilot",
    "Lab Report Intelligence",
    "ICU Command Center",
    "Audit Trail"
])

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
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "Clinical Research Copilot":

    st.markdown("<div class='card'><h3>🔬 Clinical Research Copilot</h3></div>", unsafe_allow_html=True)

    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze"):

        if mode in ["Hospital AI", "Hybrid AI"]:
            if not os.path.exists(INDEX_FILE):
                idx, docs, srcs = build_index()
            else:
                idx = faiss.read_index(INDEX_FILE)
                data = pickle.load(open(CACHE_FILE, "rb"))
                docs = data["documents"]
                srcs = data["sources"]

            qemb = embedder.encode([query])
            _, I = idx.search(np.array(qemb), 5)
            context = "\n\n".join([docs[i] for i in I[0]])

            prompt = f"""
You are a Hospital Clinical Decision Support AI.
Use ONLY hospital evidence.

Hospital Evidence:
{context}

Doctor Query:
{query}
"""
            hospital_ans = safe_ai_call(prompt)

            st.markdown("<div class='card'><h4>🏥 Hospital AI</h4></div>", unsafe_allow_html=True)
            st.write(hospital_ans)

        if mode in ["Global AI", "Hybrid AI"]:
            global_ans = safe_ai_call(query)
            st.markdown("<div class='card'><h4>🌍 Global AI</h4></div>", unsafe_allow_html=True)
            st.write(global_ans)

# ======================================================
# LAB REPORT INTELLIGENCE
# ======================================================
if module == "Lab Report Intelligence":

    st.markdown("<div class='card'><h3>🧪 Lab Report Intelligence</h3></div>", unsafe_allow_html=True)

    lab_file = st.file_uploader("Upload Lab Report PDF", type=["pdf"])

    if lab_file:
        with open("lab_report.pdf", "wb") as f:
            f.write(lab_file.getbuffer())

        reader = PdfReader("lab_report.pdf")
        text = ""
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"

        st.text_area("Extracted Report", text, height=250)

        st.markdown("<div class='card'><h4>🧠 Ask AI about this report</h4></div>", unsafe_allow_html=True)
        q = st.text_input("Ask clinical question")

        if st.button("Analyze Report"):
            prompt = f"""
You are a hospital clinical AI.

Lab Report:
{text}

Doctor Question:
{q}

Provide clinical interpretation.
"""
            ans = safe_ai_call(prompt)
            st.write(ans)

# ======================================================
# ICU COMMAND CENTER
# ======================================================
if module == "ICU Command Center":
    st.markdown("<div class='card'><h3>🚨 ICU Command Center</h3></div>", unsafe_allow_html=True)
    st.write("Live critical alerts from lab analysis will appear here.")

# ======================================================
# AUDIT TRAIL
# ======================================================
if module == "Audit Trail":
    st.markdown("<div class='card'><h3>🕒 Audit Trail</h3></div>", unsafe_allow_html=True)
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df)
    else:
        st.info("No audit records yet.")

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================

import streamlit as st
import os, json, pickle, datetime
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
USERS_DB = "users.json"
AUDIT_LOG = "audit_log.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
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
        rows = json.load(open(AUDIT_LOG))
    rows.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.get("username"),
        "event": event,
        "meta": meta or {}
    })
    json.dump(rows, open(AUDIT_LOG, "w"), indent=2)

# ======================================================
# SAFE AI WRAPPER
# ======================================================
def safe_ai_call(prompt):
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            return "⚠ AI returned empty response."
        return result["answer"]
    except Exception as e:
        audit("ai_failure", {"error": str(e)})
        return "⚠ AI service unavailable. Governance block applied."

# ======================================================
# WOW LOGIN UI (STREAMLIT SAFE)
# ======================================================
def login_ui():

    st.markdown("""
    <style>

    body {
        background: radial-gradient(circle at top, #020617 0%, #020617 60%, #000000 100%);
        font-family: 'Segoe UI', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
    }

    .login-card {
        max-width: 520px;
        margin: auto;
        margin-top: 120px;
        padding: 40px;
        border-radius: 20px;
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(20px);
        box-shadow: 0 0 80px rgba(56,189,248,0.25);
        border: 1px solid rgba(255,255,255,0.15);
        text-align: center;
    }

    .login-title {
        font-size: 36px;
        font-weight: 900;
        margin-bottom: 6px;
        background: linear-gradient(90deg,#38bdf8,#22d3ee,#0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .login-subtitle {
        color: #cbd5f5;
        font-size: 16px;
        margin-bottom: 30px;
        line-height: 1.5;
    }

    .login-footer {
        margin-top: 25px;
        font-size: 13px;
        color: #94a3b8;
    }

    .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
    }

    .stButton button {
        width: 100%;
        padding: 14px;
        border-radius: 14px;
        font-size: 17px;
        font-weight: 800;
        background: linear-gradient(90deg,#38bdf8,#22d3ee);
        color: #020617;
        border: none;
        margin-top: 15px;
        box-shadow: 0 0 30px rgba(34,211,238,0.6);
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown('<div class="login-title">ĀROGYABODHA AI</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="login-subtitle">
        Hospital Clinical Intelligence Platform<br>
        Secure Medical AI Command Center
        </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Doctor / Researcher ID")
        password = st.text_input("Secure Access Key", type="password")
        submitted = st.form_submit_button("🚀 Enter Clinical AI Platform")

    st.markdown("""
        <div class="login-footer">
        Evidence Locked • Governance Ready • ICU Intelligence
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        users = json.load(open(USERS_DB))
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = users[username]["role"]
            audit("login", {"user": username})
            st.success("✅ Secure Hospital Access Granted")
            st.rerun()
        else:
            st.error("❌ Invalid Credentials")

# ======================================================
# LOGOUT
# ======================================================
def logout_ui():
    if st.sidebar.button("Logout"):
        audit("logout", {"user": st.session_state.username})
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()

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
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, p in enumerate(reader.pages[:200]):
                t = p.extract_text()
                if t and len(t) > 100:
                    docs.append(t)
                    srcs.append(f"{pdf} — Page {i+1}")

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
# SIDEBAR
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ User: **{st.session_state.username}**")
logout_ui()

st.sidebar.subheader("📁 Hospital Evidence Library")

uploads = st.sidebar.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Evidence Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    st.sidebar.success("Hospital Evidence Index Built")

st.sidebar.markdown("🟢 Index Status: READY" if os.path.exists(INDEX_FILE) else "🔴 Index Status: NOT BUILT")

module = st.sidebar.radio("Select Module", [
    "Clinical Research Copilot",
    "Audit Trail"
])

# ======================================================
# HEADER
# ======================================================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform")
st.caption("Hospital-grade • Evidence-locked • Governance enabled")

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "Clinical Research Copilot":
    st.subheader("🔬 Clinical Research Copilot")

    query = st.text_input("Ask a clinical research question")

    if st.button("🚀 Analyze") and query:
        audit("clinical_query", {"query": query})

        if not st.session_state.index_ready:
            st.error("Hospital evidence index not built.")
        else:
            qemb = embedder.encode([query])
            _, I = st.session_state.index.search(np.array(qemb), 5)

            context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
            sources = [st.session_state.sources[i] for i in I[0]]

            prompt = f"Use only hospital evidence:\n{context}\n\nQuestion:{query}"
            answer = safe_ai_call(prompt)

            st.success("Hospital Evidence Answer")
            st.write(answer)

            st.markdown("### 📑 Evidence Sources")
            for s in sources:
                st.info(s)

# ======================================================
# AUDIT TRAIL
# ======================================================
if module == "Audit Trail":
    st.subheader("🕒 Audit Trail")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

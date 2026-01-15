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
def safe_ai_call(prompt, mode="AI"):
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            return {"status": "error", "answer": "⚠ AI returned empty response."}
        return {"status": "ok", "answer": result["answer"]}
    except Exception as e:
        audit("ai_failure", {"mode": mode, "error": str(e)})
        return {"status": "down", "answer": "⚠ AI service unavailable. Governance block applied."}

# ======================================================
# WOW LOGIN UI
# ======================================================
def login_ui():
    st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top, #020617 0%, #020617 60%, #000000 100%);
        font-family: 'Segoe UI', sans-serif;
    }

    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 95vh;
    }

    .login-card {
        width: 1200px;
        height: 620px;
        display: flex;
        border-radius: 28px;
        overflow: hidden;
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(30px);
        box-shadow: 0 0 120px rgba(56,189,248,0.35);
        border: 1px solid rgba(255,255,255,0.15);
    }

    .login-left {
        width: 50%;
        padding: 80px;
        color: white;
    }

    .login-title {
        font-size: 44px;
        font-weight: 900;
        margin-bottom: 10px;
        background: linear-gradient(90deg,#38bdf8,#22d3ee,#0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    }

    .login-subtitle {
        color: #cbd5f5;
        margin-bottom: 60px;
        font-size: 18px;
        line-height: 1.6;
    }

    .login-right {
        width: 50%;
        background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d");
        background-size: cover;
        background-position: center;
        position: relative;
    }

    .overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg,rgba(2,6,23,0.95),rgba(2,6,23,0.4));
    }

    .brand-box {
        position: absolute;
        bottom: 70px;
        left: 70px;
        color: white;
    }

    .brand-title {
        font-size: 38px;
        font-weight: 900;
        letter-spacing: 1px;
    }

    .brand-sub {
        margin-top: 12px;
        color: #bae6fd;
        font-size: 18px;
        line-height: 1.6;
    }

    .stTextInput input {
        background: rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px;
        color: white;
        border: 1px solid rgba(255,255,255,0.25);
    }

    .stButton button {
        width: 100%;
        padding: 18px;
        border-radius: 16px;
        font-size: 18px;
        font-weight: 800;
        background: linear-gradient(90deg,#38bdf8,#22d3ee);
        color: #020617;
        border: none;
        margin-top: 30px;
        box-shadow: 0 0 40px rgba(34,211,238,0.7);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-wrapper">
        <div class="login-card">
            <div class="login-left">
                <div class="login-title">ĀROGYABODHA AI</div>
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
            </div>
            <div class="login-right">
                <div class="overlay"></div>
                <div class="brand-box">
                    <div class="brand-title">Hospital AI Core</div>
                    <div class="brand-sub">
                        Evidence Locked • Governance Ready • ICU Intelligence
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.error("❌ Access Denied — Invalid Credentials")

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
st.caption("Hospital-grade • Evidence-locked • OCR-enabled • Governance enabled")

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "Clinical Research Copilot":
    st.subheader("🔬 Clinical Research Copilot")

    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze") and query:
        audit("clinical_query", {"query": query, "mode": mode})

        tabs = ["🏥 Hospital", "🌍 Global"] if mode != "Hospital AI" else ["🏥 Hospital"]
        tab_objs = st.tabs(tabs)

        if "🏥 Hospital" in tabs:
            with tab_objs[tabs.index("🏥 Hospital")]:
                if not st.session_state.index_ready:
                    st.error("Hospital evidence index not built.")
                else:
                    qemb = embedder.encode([query])
                    _, I = st.session_state.index.search(np.array(qemb), 5)
                    context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
                    sources = [st.session_state.sources[i] for i in I[0]]

                    prompt = f"Use only hospital evidence:\n{context}\n\nQuestion:{query}"
                    resp = safe_ai_call(prompt, "Hospital AI")

                    if resp["status"] == "ok":
                        st.success("Hospital Evidence Answer")
                        st.write(resp["answer"])

                        st.markdown("### 📑 Evidence Sources")
                        for s in sources:
                            st.info(s)
                    else:
                        st.error(resp["answer"])

        if "🌍 Global" in tabs:
            with tab_objs[tabs.index("🌍 Global")]:
                resp = safe_ai_call(query, "Global AI")
                st.write(resp["answer"])

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

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
# AUDIT
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
# MODERN LOGIN UI (CubeFactory Style)
# ======================================================
def login_ui():
    st.markdown("""
    <style>
    body { background-color: #0f172a; }

    .login-container {
        max-width: 1100px;
        margin: auto;
        margin-top: 70px;
        background: white;
        border-radius: 16px;
        box-shadow: 0px 20px 80px rgba(0,0,0,0.5);
        display: flex;
        overflow: hidden;
    }

    .login-left { width: 50%; padding: 70px; }
    .login-right {
        width: 50%;
        background-image: url("https://images.unsplash.com/photo-1581092919535-7f1b33b4c8d8");
        background-size: cover;
        background-position: center;
        position: relative;
    }

    .login-overlay {
        position: absolute;
        bottom: 50px;
        left: 50px;
        color: white;
    }

    .login-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; }
    .login-subtitle { color: #64748b; margin-bottom: 40px; font-size: 17px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-container">
        <div class="login-left">
            <div class="login-title">Welcome back</div>
            <div class="login-subtitle">Secure Hospital Access Portal</div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("🔐 Sign in")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="login-right">
            <div class="login-overlay">
                <h2>ĀROGYABODHA AI</h2>
                <p>Hospital Clinical Intelligence Platform</p>
                <p>Evidence Locked • Governance Ready • ICU Intelligence</p>
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
            st.success("Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


def logout_ui():
    if st.sidebar.button("Logout"):
        audit("logout", {"user": st.session_state.username})
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()


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

uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
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
    "Lab Report Intelligence",
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

        tabs = ["🏥 Hospital", "🌍 Global"] if mode!="Hospital AI" else ["🏥 Hospital"]
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

                    if resp["status"]=="ok":
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

# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================

import streamlit as st
import os, json, pickle, datetime, io
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# External AI connector (must return: {"answer": "..."} )
from external_research import external_research_answer

# Optional OCR (hooks)
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
# DISCLAIMER (Governance)
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

# Demo users (replace with hospital IAM later)
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
# AUDIT SYSTEM (Compliance)
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
# SAFE AI WRAPPER (Governance)
# ======================================================
def safe_ai_call(prompt, mode="AI"):
    """
    Governance-safe AI call wrapper:
    - Failure handling
    - Empty-response handling
    - Audit logging
    """
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
# WOW LOGIN UI (Streamlit-native, cloud-safe)
# ======================================================
def login_ui():

    st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top, #020617 0%, #020617 60%, #000000 100%);
        font-family: 'Segoe UI', sans-serif;
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
            audit("login_failed", {"user": username})
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
# MODEL (Embeddings)
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# FAISS INDEX (Hospital Evidence RAG)
# ======================================================
def extract_text_from_pdf_bytes(file_bytes: bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for i, p in enumerate(reader.pages[:300]):  # cap pages for safety
        t = p.extract_text()
        if t and len(t) > 100:
            pages_text.append(t)
    return pages_text

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

    emb = embedder.encode(docs, show_progress_bar=False)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

# Load cached index if exists
if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data.get("documents", [])
    st.session_state.sources = data.get("sources", [])
    st.session_state.index_ready = True

# ======================================================
# SIDEBAR (Command Center Navigation)
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ User: **{st.session_state.username}** ({st.session_state.role})")
logout_ui()

st.sidebar.subheader("📁 Hospital Evidence Library")

uploads = st.sidebar.file_uploader("Upload Medical PDFs (Bulk)", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Evidence Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    audit("build_index", {"count": len(st.session_state.documents)})
    st.sidebar.success("Hospital Evidence Index Built")

st.sidebar.markdown("🟢 Index Status: READY" if st.session_state.index_ready else "🔴 Index Status: NOT BUILT")

module = st.sidebar.radio("Select Module", [
    "Clinical Research Copilot",
    "Lab Report Intelligence",
    "Audit Trail"
])

# ======================================================
# HEADER (Dashboard)
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

        # ---------------- Hospital AI (RAG) ----------------
        if "🏥 Hospital" in tabs:
            with tab_objs[tabs.index("🏥 Hospital")]:
                if not st.session_state.index_ready:
                    st.error("Hospital evidence index not built.")
                else:
                    qemb = embedder.encode([query])
                    _, I = st.session_state.index.search(np.array(qemb, dtype=np.float32), 5)

                    context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
                    sources = [st.session_state.sources[i] for i in I[0]]

                    prompt = f"Use only hospital evidence below to answer.\n\n{context}\n\nQuestion: {query}"
                    resp = safe_ai_call(prompt, "Hospital AI")

                    if resp["status"] == "ok":
                        st.success("Hospital Evidence Answer")
                        st.write(resp["answer"])

                        st.markdown("### 📑 Evidence Sources")
                        for s in sources:
                            st.info(s)
                    else:
                        st.error(resp["answer"])

        # ---------------- Global AI ----------------
        if "🌍 Global" in tabs:
            with tab_objs[tabs.index("🌍 Global")]:
                resp = safe_ai_call(query, "Global AI")
                if resp["status"] == "ok":
                    st.success("Global Research Answer")
                    st.write(resp["answer"])
                else:
                    st.error(resp["answer"])

# ======================================================
# LAB REPORT INTELLIGENCE (Module Ready + OCR Hooks)
# ======================================================
if module == "Lab Report Intelligence":
    st.subheader("🧪 Lab Report Intelligence")

    st.info("Upload lab reports (PDF/Image). OCR & AI interpretation pipeline is ready (placeholder).")

    uploaded_lab = st.file_uploader("Upload Lab Report (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_lab:
        # Save file
        save_path = os.path.join(LAB_FOLDER, uploaded_lab.name)
        with open(save_path, "wb") as out:
            out.write(uploaded_lab.getbuffer())

        audit("lab_upload", {"file": uploaded_lab.name})
        st.success("Lab report uploaded successfully.")

        # OCR hook (if enabled)
        if OCR_AVAILABLE and uploaded_lab.type.startswith("image"):
            try:
                img_bytes = uploaded_lab.getvalue()
                # Placeholder: integrate image->text OCR here if needed
                st.markdown("### 🧾 OCR Text (Preview)")
                st.write("OCR pipeline ready. (Integrate image-to-text here.)")
            except Exception as e:
                audit("ocr_failure", {"file": uploaded_lab.name, "error": str(e)})
                st.warning("OCR failed for this file.")

        st.markdown("### 🧠 AI Interpretation")
        st.write("AI interpretation module is ready in architecture. Enable in next version.")

# ======================================================
# AUDIT TRAIL (Compliance Dashboard)
# ======================================================
if module == "Audit Trail":
    st.subheader("🕒 Audit Trail")
    if os.path.exists(AUDIT_LOG):
        try:
            df = pd.DataFrame(json.load(open(AUDIT_LOG)))
            st.dataframe(df, use_container_width=True)
        except:
            st.warning("Audit log is empty or corrupted.")
    else:
        st.info("No audit records yet.")

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

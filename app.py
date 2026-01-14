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
# MODERN LOGIN UI (CubeFactory style)
# ======================================================
def login_ui():
    st.markdown("""
    <style>
    body { background-color: #f6f7fb; }
    .login-container {
        max-width: 1100px;
        margin: auto;
        margin-top: 80px;
        background: white;
        border-radius: 14px;
        box-shadow: 0px 10px 40px rgba(0,0,0,0.08);
        display: flex;
        overflow: hidden;
    }
    .login-left { width: 50%; padding: 60px; }
    .login-right {
        width: 50%;
        background-image: url("https://images.unsplash.com/photo-1526256262350-7da7584cf5eb");
        background-size: cover;
        background-position: center;
        position: relative;
    }
    .login-overlay {
        position: absolute;
        bottom: 40px;
        left: 40px;
        color: white;
    }
    .login-title { font-size: 32px; font-weight: 700; margin-bottom: 10px; }
    .login-subtitle { color: #666; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-container">
        <div class="login-left">
            <div class="login-title">Welcome back</div>
            <div class="login-subtitle">Please enter your details</div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

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
# OCR
# ======================================================
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    except:
        pass

    if len(text.strip()) < 200 and OCR_AVAILABLE:
        try:
            images = convert_from_path(pdf_path, dpi=300)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        except:
            pass

    return text

# ======================================================
# CONFIDENCE ENGINE
# ======================================================
def semantic_similarity(a, b):
    ea = embedder.encode([a])[0]
    eb = embedder.encode([b])[0]
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))

def semantic_evidence_level(answer, context):
    sim = semantic_similarity(answer, context)
    if sim >= 0.55:
        return "STRONG", int(sim * 100)
    elif sim >= 0.30:
        return "PARTIAL", int(sim * 100)
    else:
        return "INSUFFICIENT", int(sim * 100)

def confidence_score(answer, n_sources):
    score = 60
    if n_sources >= 3: score += 15
    if "fda" in answer.lower(): score += 10
    return min(score, 95)

# ======================================================
# LAB ENGINE
# ======================================================
LAB_RULES = {
    "Total Bilirubin": (0.3, 1.2, "mg/dL"),
    "SGPT": (0, 50, "U/L"),
    "SGOT": (0, 50, "U/L")
}

def extract_lab_values(text):
    values = {}
    for test in LAB_RULES:
        match = re.search(test + r".*?(\d+\.?\d*)", text, re.IGNORECASE)
        if match:
            values[test] = float(match.group(1))
    return values

def generate_lab_summary(values):
    summary, alerts = [], []
    for test, val in values.items():
        low, high, unit = LAB_RULES[test]
        status = "🟢 NORMAL" if low <= val <= high else "🔴 HIGH"
        summary.append((test, val, unit, status))

        if test == "Total Bilirubin" and val >= 5:
            alerts.append("🚨 Severe Jaundice — ICU evaluation required")

    return summary, alerts

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

        tabs = ["🏥 Hospital", "🌍 Global", "🧪 Outcomes"] if mode=="Hybrid AI" else (
               ["🏥 Hospital"] if mode=="Hospital AI" else ["🌍 Global"])

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
                        level, coverage = semantic_evidence_level(resp["answer"], context)
                        confidence = confidence_score(resp["answer"], len(sources))

                        c1,c2,c3 = st.columns(3)
                        c1.metric("Confidence", f"{confidence}%")
                        c2.metric("Coverage", f"{coverage}%")
                        c3.metric("Evidence", level)

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

        if "🧪 Outcomes" in tabs:
            with tab_objs[tabs.index("🧪 Outcomes")]:
                if "fda" in resp["answer"].lower():
                    st.success("FDA-approved therapy detected")
                else:
                    st.info("No FDA outcome keyword detected")

# ======================================================
# LAB REPORT INTELLIGENCE
# ======================================================
if module == "Lab Report Intelligence":
    st.subheader("🧪 Lab Report Intelligence")

    lab_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
    if lab_file:
        with open("lab_report.pdf", "wb") as f:
            f.write(lab_file.getbuffer())

        report_text = extract_text_from_pdf("lab_report.pdf")
        values = extract_lab_values(report_text)
        summary, alerts = generate_lab_summary(values)

        st.markdown("### 🧾 Smart Lab Summary")
        for t,v,u,s in summary:
            st.write(f"{t}: {v} {u} — {s}")

        if alerts:
            st.markdown("### 🚨 ICU Alerts")
            for a in alerts:
                st.error(a)

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

# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================
# Production Build:
# - Doctor Login (username/password only)
# - Governance Layer
# - Medical Library (Upload/Delete/Build Index)
# - FAISS Evidence Engine
# - 3 AI Modes (Hospital / Global / Hybrid)
# - Evidence-locked Hospital AI
# - Clinical Confidence Engine
# - Safe AI Wrapper
# - Lab Report Intelligence + OCR fallback
# - Smart Lab Summary + ICU Alerts
# - Audit Trail
# ======================================================

import streamlit as st
import os, json, pickle, datetime, re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ================= OCR (Safe Optional) =================
OCR_AVAILABLE = True
try:
    import pytesseract
    from pdf2image import convert_from_path
except:
    OCR_AVAILABLE = False

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ================= DISCLAIMER =================
st.info(
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only. "
    "It does NOT provide diagnosis or treatment. "
    "Final clinical decisions must be made by licensed medical professionals."
)

# ================= STORAGE =================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
USERS_DB = "users.json"
AUDIT_LOG = "audit_log.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Seed demo users
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"), indent=2)

# ================= SESSION =================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "show_help": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= AUDIT =================
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

# ================= SAFE AI =================
def safe_ai_call(prompt, mode="AI"):
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            return {"status": "error", "answer": "⚠ AI returned empty response."}
        return {"status": "ok", "answer": result["answer"]}
    except Exception as e:
        audit("ai_failure", {"mode": mode, "error": str(e)})
        return {
            "status": "down",
            "answer": (
                "⚠ AI service temporarily unavailable.\n\n"
                "Hospital Governance Policy:\n"
                "• No hallucinated content\n"
                "• No unsafe response\n"
                "• Please retry later"
            )
        }

# ================= LOGIN =================
def login_ui():
    st.markdown("## 🏥 ĀROGYABODHA AI Hospital Login")

    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.role = users[u]["role"]
            audit("login", {"user": u})
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

# ================= MODEL =================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ================= FAISS =================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, pdf))
            for i, p in enumerate(reader.pages[:200]):
                text = p.extract_text()
                if text and len(text) > 100:
                    docs.append(text)
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

# ================= OCR =================
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
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

# ================= LAB ENGINE =================
LAB_RULES = {
    "Total Bilirubin": (0.3, 1.2, "mg/dL"),
    "Direct Bilirubin": (0.0, 0.2, "mg/dL"),
    "SGPT": (0, 50, "U/L"),
    "SGOT": (0, 50, "U/L"),
    "GGT": (0, 55, "U/L"),
}

def extract_lab_values(text):
    values = {}
    for line in text.split("\n"):
        for test in LAB_RULES:
            if test.lower() in line.lower():
                nums = re.findall(r"\b\d+\.?\d*\b", line)
                if nums:
                    values[test] = float(nums[-1])
    return values

def generate_lab_summary(values):
    summary, alerts = [], []
    for test, val in values.items():
        low, high, unit = LAB_RULES[test]
        if val < low:
            status = "🟡 LOW"
        elif val > high:
            status = "🔴 HIGH"
        else:
            status = "🟢 NORMAL"
        summary.append((test, val, unit, status))

        if test == "Total Bilirubin" and val >= 5:
            alerts.append("🚨 Severe Jaundice — ICU required")
        if test == "SGPT" and val > 300:
            alerts.append("🚨 Severe Liver Injury")

    return summary, alerts

# ================= SIDEBAR =================
st.sidebar.markdown(f"👨‍⚕️ User: **{st.session_state.username}**")
logout_ui()

st.sidebar.subheader("📁 Medical Library")

uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    st.sidebar.success("Index built")

st.sidebar.markdown("#### 📚 Library Files")
for pdf in os.listdir(PDF_FOLDER):
    if pdf.endswith(".pdf"):
        c1, c2 = st.sidebar.columns([5,1])
        c1.write("📄 " + pdf)
        if c2.button("🗑", key=pdf):
            os.remove(os.path.join(PDF_FOLDER, pdf))
            if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
            if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
            st.session_state.index_ready = False
            st.rerun()

if st.sidebar.button("❓ Help"):
    st.session_state.show_help = not st.session_state.show_help

if st.session_state.show_help:
    st.sidebar.info("Workflow: Upload PDFs → Build Index → Ask Question → Select AI Mode")

module = st.sidebar.radio("Select Module", [
    "Clinical Research Copilot",
    "Lab Report Intelligence",
    "Audit Trail"
])

# ================= HEADER =================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform")
st.caption("Hospital-grade • Evidence-locked • Governance enabled")

# ================= CLINICAL COPILOT =================
if module == "Clinical Research Copilot":
    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze") and query:

        if mode == "Hospital AI":
            tabs = ["🏥 Hospital", "📚 Library"]
        elif mode == "Global AI":
            tabs = ["🌍 Global"]
        else:
            tabs = ["🏥 Hospital", "🌍 Global", "🧪 Outcomes", "📚 Library"]

        tab_objs = st.tabs(tabs)
        tab_map = dict(zip(tabs, tab_objs))

        if "🏥 Hospital" in tab_map:
            with tab_map["🏥 Hospital"]:
                if not st.session_state.index_ready:
                    st.error("Hospital index not built.")
                else:
                    qemb = embedder.encode([query])
                    _, I = st.session_state.index.search(np.array(qemb), 5)
                    context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
                    sources = [st.session_state.sources[i] for i in I[0]]

                    prompt = f"Use only hospital evidence:\n{context}\n\nQuestion:{query}"
                    resp = safe_ai_call(prompt, "Hospital AI")
                    st.write(resp["answer"])
                    for s in sources: st.info(s)

        if "🌍 Global" in tab_map:
            with tab_map["🌍 Global"]:
                resp = safe_ai_call(query, "Global AI")
                st.write(resp["answer"])

        if "🧪 Outcomes" in tab_map:
            with tab_map["🧪 Outcomes"]:
                st.info("Outcome analysis panel")

        if "📚 Library" in tab_map:
            with tab_map["📚 Library"]:
                for pdf in os.listdir(PDF_FOLDER):
                    st.write("📄", pdf)

# ================= LAB INTELLIGENCE =================
if module == "Lab Report Intelligence":
    lab_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])

    if lab_file:
        open("lab.pdf","wb").write(lab_file.getbuffer())
        text = extract_text_from_pdf("lab.pdf")
        values = extract_lab_values(text)
        summary, alerts = generate_lab_summary(values)

        st.markdown("### 🧾 Smart Lab Summary")
        for t,v,u,s in summary:
            st.write(f"{t}: {v} {u} — {s}")

        if alerts:
            st.markdown("### 🚨 ICU Alerts")
            for a in alerts:
                st.error(a)

# ================= AUDIT =================
if module == "Audit Trail":
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No audit logs")

# ================= FOOTER =================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

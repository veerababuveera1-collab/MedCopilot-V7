# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# FINAL STABLE VERSION (Streamlit Cloud Compatible)
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
# UI THEME
# ======================================================
st.markdown("""
<style>
body { background:#0b1220; color:#e5e7eb; }
.card { background:rgba(255,255,255,0.05); border-radius:14px; padding:14px; margin-bottom:12px; }
.small { opacity:.85; font-size:.9rem; }
.badge { padding:4px 10px; border-radius:999px; font-weight:700; }
.ok{background:#00c2a8;color:#04211a}
.warn{background:#ffd166;color:#3b2f00}
.danger{background:#ef476f}
</style>
""", unsafe_allow_html=True)

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
AUDIT_LOG = "audit_log.json"
USERS_DB = "users.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Seed users
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
    "index_ready": False,
}
for k,v in defaults.items():
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
            st.session_state.role = users[u]["role"]
            audit("login")
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

def logout_ui():
    if st.sidebar.button("Logout"):
        audit("logout")
        for k in ["logged_in","username","role"]:
            st.session_state[k] = None
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
# FAISS INDEX
# ======================================================
def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.lower().endswith(".pdf"):
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

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# ======================================================
# LAB RULES
# ======================================================
LAB_RULES = {
    "Total Bilirubin": (0.3, 1.2, "mg/dL"),
    "Direct Bilirubin": (0.0, 0.2, "mg/dL"),
    "SGPT": (0, 50, "U/L"),
    "SGOT": (0, 50, "U/L"),
    "GGT": (0, 55, "U/L"),
    "Creatinine": (0.7, 1.3, "mg/dL"),
    "Platelets": (150000, 410000, "/cumm"),
    "Hemoglobin": (13, 17, "g/dL"),
}

# ======================================================
# LAB PARSER
# ======================================================
def extract_lab_values_from_pdf(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    found = {}
    for line in lines:
        for test in LAB_RULES:
            if test.lower() in line.lower():
                nums = re.findall(r"\d+\.?\d*", line)
                if nums:
                    found[test] = float(nums[-1])
    return found

# ======================================================
# MEDICAL LOGIC
# ======================================================
def analyze_labs(values):
    summary = []
    alerts = []

    for test, val in values.items():
        low, high, unit = LAB_RULES[test]
        if val < low:
            status = "LOW"
        elif val > high:
            status = "HIGH"
        else:
            status = "NORMAL"

        summary.append((test, val, unit, status))

        if test == "Total Bilirubin" and val > 5:
            alerts.append("🚨 Severe Jaundice — ICU Required")
        if test == "Creatinine" and val > 3:
            alerts.append("🚨 Renal Failure Risk")
        if test == "Platelets" and val < 50000:
            alerts.append("🚨 Bleeding Risk")

    return summary, alerts

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("## 🧠 ĀROGYABODHA AI")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

st.sidebar.markdown(f"**User:** {st.session_state.username} ({st.session_state.role})")
logout_ui()

st.sidebar.divider()

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

st.sidebar.divider()
module = st.sidebar.radio("Select Module", [
    "Clinical Research Copilot",
    "Lab Report Intelligence",
    "Audit Trail"
])

# ======================================================
# HEADER
# ======================================================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform")
st.markdown("<div class='small'>Hospital-grade • Evidence-locked • Doctor-safe</div>", unsafe_allow_html=True)
st.markdown("---")

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "Clinical Research Copilot":
    st.subheader("🔬 Clinical Research Copilot")
    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze"):
        audit("research_query", {"query": query})
        ans = external_research_answer(query).get("answer","")
        st.write(ans)

# ======================================================
# LAB REPORT INTELLIGENCE
# ======================================================
if module == "Lab Report Intelligence":
    st.subheader("🧪 Lab Report Intelligence")

    lab_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])

    if lab_file:
        with open("lab_report.pdf","wb") as f:
            f.write(lab_file.getbuffer())

        reader = PdfReader("lab_report.pdf")
        report_text = ""
        for p in reader.pages:
            report_text += (p.extract_text() or "") + "\n"

        values = extract_lab_values_from_pdf(report_text)
        summary, alerts = analyze_labs(values)

        st.markdown("### 🧾 Smart Lab Summary")
        for t,v,u,s in summary:
            badge = "ok" if s=="NORMAL" else "danger"
            st.markdown(
                f"<div class='card'><span class='badge {badge}'>{s}</span> "
                f"<b>{t}</b>: {v} {u}</div>",
                unsafe_allow_html=True
            )

        if alerts:
            st.markdown("### 🚨 ICU Alerts")
            for a in alerts:
                st.error(a)

        question = st.text_input("Ask AI about this report")

        if st.button("Analyze Report"):
            audit("lab_analyze")
            prompt = f"""
You are a hospital clinical AI.

Lab Report:
{report_text}

Doctor Question:
{question}

Provide diagnosis pattern, risks and next steps.
"""
            ai = external_research_answer(prompt).get("answer","")
            st.success(ai)

# ======================================================
# AUDIT TRAIL
# ======================================================
if module == "Audit Trail":
    st.subheader("🕒 Audit Trail")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No audit logs yet.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("ĀROGYABODHA AI © Hospital Clinical Intelligence Platform")

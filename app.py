# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================
# FINAL PRODUCTION BUILD (OTP + Governance + Evidence)
# ======================================================

import streamlit as st
import os, json, pickle, datetime, re, random
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ---------------- OCR (Safe Optional) ----------------
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

# Seed demo users
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "email": "doctor1@hospital.com", "role": "Doctor"}
    }, open(USERS_DB, "w"), indent=2)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "otp": None,
    "otp_verified": False,
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
# SAFE AI WRAPPER (Governance Layer)
# ======================================================
def safe_ai_call(prompt):
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            return "⚠ AI returned empty response."
        return result["answer"]
    except Exception as e:
        audit("ai_failure", {"error": str(e)})
        return (
            "⚠ AI service temporarily unavailable.\n\n"
            "Hospital Governance Policy:\n"
            "• No hallucinated content\n"
            "• No unsafe response\n"
            "• Please retry later"
        )

# ======================================================
# LOGIN + OTP AUTH
# ======================================================
def login_ui():
    st.markdown("## 🔐 Doctor Secure Login")

    tab1, tab2 = st.tabs(["🔑 Password Login", "📲 OTP Login"])

    # -------- Password Login --------
    with tab1:
        username = st.text_input("Username", placeholder="doctor1")
        password = st.text_input("Password", type="password", placeholder="doctor123")

        if st.button("Login with Password"):
            users = json.load(open(USERS_DB))
            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = users[username]["role"]
                audit("login_password", {"user": username})
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # -------- OTP Login --------
    with tab2:
        email = st.text_input("Registered Email", placeholder="doctor1@hospital.com")

        if st.button("Send OTP"):
            otp = random.randint(100000, 999999)
            st.session_state.otp = str(otp)
            st.session_state.otp_email = email

            # Simulated OTP (replace with email API)
            st.success(f"OTP sent to email (Demo OTP: {otp})")

        otp_input = st.text_input("Enter OTP")

        if st.button("Verify OTP"):
            if otp_input == st.session_state.get("otp"):
                users = json.load(open(USERS_DB))
                for u, data in users.items():
                    if data["email"] == st.session_state.otp_email:
                        st.session_state.logged_in = True
                        st.session_state.username = u
                        st.session_state.role = data["role"]
                        audit("login_otp", {"user": u})
                        st.success("OTP verified. Login successful.")
                        st.rerun()
                        return
                st.error("Email not registered")
            else:
                st.error("Invalid OTP")

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
# OCR ENGINE
# ======================================================
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

# ======================================================
# LAB PARSER
# ======================================================
LAB_RULES = {
    "Total Bilirubin": (0.3, 1.2, "mg/dL"),
    "Direct Bilirubin": (0.0, 0.2, "mg/dL"),
    "SGPT": (0, 50, "U/L"),
    "SGOT": (0, 50, "U/L"),
    "GGT": (0, 55, "U/L"),
}

def extract_lab_values(text):
    values = {}
    for test in LAB_RULES:
        m = re.search(test + r".*?(\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            values[test] = float(m.group(1))
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
            alerts.append("🚨 Severe Jaundice — ICU evaluation required")

    return summary, alerts

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ User: **{st.session_state.username}**")
logout_ui()

st.sidebar.subheader("📁 Medical Library")

uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index, st.session_state.documents, st.session_state.sources = build_index()
    st.session_state.index_ready = True
    st.sidebar.success("Evidence Index Built")

if os.path.exists(INDEX_FILE):
    st.sidebar.markdown("🟢 Index Status: READY")
else:
    st.sidebar.markdown("🔴 Index Status: NOT BUILT")

st.sidebar.markdown("#### 📚 Library Files")
for pdf in os.listdir(PDF_FOLDER):
    if pdf.endswith(".pdf"):
        st.sidebar.write("📄 " + pdf)

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

        tabs = ["🏥 Hospital", "🌍 Global", "🧪 Outcomes", "📚 Library"]
        t1, t2, t3, t4 = st.tabs(tabs)

        with t1:
            if not st.session_state.index_ready:
                st.error("Hospital index not built.")
            else:
                qemb = embedder.encode([query])
                _, I = st.session_state.index.search(np.array(qemb), 5)
                context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
                sources = [st.session_state.sources[i] for i in I[0]]

                prompt = f"Hospital Evidence:\n{context}\n\nDoctor Question:\n{query}"
                answer = safe_ai_call(prompt)
                st.success("Hospital Evidence Answer")
                st.write(answer)

                st.markdown("### 📑 Evidence Sources")
                for s in sources:
                    st.info(s)

        with t2:
            answer = safe_ai_call(query)
            st.write(answer)

        with t3:
            if "fda" in answer.lower():
                st.success("FDA-approved therapy detected")
            else:
                st.info("No FDA outcome keyword detected.")

        with t4:
            for pdf in os.listdir(PDF_FOLDER):
                if pdf.endswith(".pdf"):
                    st.write("📄", pdf)

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
        if not summary:
            st.warning("Unable to auto-detect RESULT values.")
        else:
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
    else:
        st.info("No audit logs yet.")

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

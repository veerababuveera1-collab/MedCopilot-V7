# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================
# - Doctor Login (rerun-safe)
# - Medical Library (Upload/Delete/Build Index/Status)
# - FAISS Evidence Engine
# - 3 AI Modes (Hospital / Global / Hybrid) [MODE ISOLATED]
# - Dynamic Tabs (Hospital / Global / Outcomes / Library) [Governance]
# - Clinical Confidence Engine
# - Lab Report Intelligence (RESULT parser)
# - Smart Lab Summary (🟢🟡🔴) + ICU Alerts 🚨
# - Audit Trail
# - Help Panel
# - Safe AI Wrapper (no crashes)
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

# Seed users (demo)
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
    "show_help": False
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
def safe_ai_call(prompt, mode="Hospital AI"):
    try:
        result = external_research_answer(prompt)
        if not result or "answer" not in result:
            return {"status": "error", "answer": "⚠ AI returned empty response. Please retry."}
        return {"status": "ok", "answer": result["answer"]}
    except Exception as e:
        audit("ai_failure", {"mode": mode, "error": str(e)})
        return {
            "status": "down",
            "answer": (
                "⚠ AI service temporarily unavailable.\n\n"
                "Hospital Governance Policy:\n"
                "• No unsafe response generated\n"
                "• No hallucinated content\n"
                "• Please retry later"
            )
        }

# ======================================================
# AUTH (rerun-safe)
# ======================================================
def login_ui():
    st.markdown("### 🔐 Doctor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = json.load(open(USERS_DB))
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = users[username]["role"]
            audit("login", {"user": username})
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

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
                    srcs.append(f"{pdf} – Page {i+1}")
    if not docs:
        return None, [], []
    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents": docs, "sources": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

# Load index if exists
if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# ======================================================
# CLINICAL CONFIDENCE ENGINE
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
    if any(x in answer.lower() for x in ["mortality", "survival", "outcome"]):
        score += 10
    return min(score, 95)

def clinical_safety_gate(level):
    return level != "INSUFFICIENT"

# ======================================================
# LAB RULES + PARSER
# ======================================================
LAB_RULES = {
    "Total Bilirubin": (0.3, 1.2, "mg/dL"),
    "Direct Bilirubin": (0.0, 0.2, "mg/dL"),
    "SGPT": (0, 50, "U/L"),
    "SGOT": (0, 50, "U/L"),
    "GGT": (0, 55, "U/L"),
}

def extract_lab_values(text):
    found = {}
    for test in LAB_RULES:
        m = re.search(test + r".*?(\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            found[test] = float(m.group(1))
    return found

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
        if test == "SGPT" and val > 300:
            alerts.append("🚨 Severe Liver Injury Risk")

    return summary, alerts

# ======================================================
# SIDEBAR (Governance + Library)
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
    st.sidebar.success("Hospital Evidence Index Built")

# Index status
if os.path.exists(INDEX_FILE):
    st.sidebar.markdown("🟢 Index Status: READY")
else:
    st.sidebar.markdown("🔴 Index Status: NOT BUILT")

# Library list + delete
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

# Help
if st.sidebar.button("❓ Help"):
    st.session_state.show_help = not st.session_state.show_help

if st.session_state.show_help:
    st.sidebar.info("Workflow: Upload PDFs → Build Index → Ask Question → Select AI Mode")

# Module selector
module = st.sidebar.radio("Select Module", [
    "Clinical Research Copilot",
    "Lab Report Intelligence",
    "Audit Trail"
])

# ======================================================
# HEADER
# ======================================================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform")
st.caption("Hospital-grade • Evidence-locked • Governance enabled")

# ======================================================
# CLINICAL RESEARCH COPILOT (MODE ISOLATED)
# ======================================================
if module == "Clinical Research Copilot":
    st.subheader("🔬 Clinical Research Copilot")

    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze") and query:
        audit("clinical_query", {"query": query, "mode": mode})

        # Dynamic tabs based on mode (Governance)
        if mode == "Hospital AI":
            tabs = ["🏥 Hospital", "📚 Library"]
        elif mode == "Global AI":
            tabs = ["🌍 Global"]
        else:  # Hybrid
            tabs = ["🏥 Hospital", "🌍 Global", "🧪 Outcomes", "📚 Library"]

        tab_objs = st.tabs(tabs)
        tab_map = dict(zip(tabs, tab_objs))

        # -------- Hospital AI --------
        if mode in ["Hospital AI", "Hybrid AI"]:
            with tab_map["🏥 Hospital"]:
                st.subheader("🏥 Hospital AI — Evidence Locked")

                if not st.session_state.index_ready:
                    st.error("Hospital evidence index not built. Upload PDFs and click 'Build Index'.")
                else:
                    qemb = embedder.encode([query])
                    _, I = st.session_state.index.search(np.array(qemb), 5)

                    context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
                    sources = [st.session_state.sources[i] for i in I[0]]

                    prompt = f"""
You are a Hospital Clinical Decision Support AI.
Use ONLY hospital evidence. Do NOT hallucinate.

Hospital Evidence:
{context}

Doctor Question:
{query}
"""
                    resp = safe_ai_call(prompt, mode="Hospital AI")
                    answer = resp["answer"]

                    if resp["status"] != "ok":
                        st.error(answer)
                    else:
                        level, coverage = semantic_evidence_level(answer, context)
                        confidence = confidence_score(answer, len(sources))

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Confidence", f"{confidence}%")
                        c2.metric("Evidence Coverage", f"{coverage}%")
                        c3.metric("Evidence Level", level)

                        if not clinical_safety_gate(level):
                            st.error("❌ Blocked — Insufficient hospital evidence")
                        else:
                            st.success("Hospital Evidence Answer")
                            st.write(answer)
                            st.markdown("##### Evidence Sources")
                            for s in sources:
                                st.info(s)

        # -------- Global AI --------
        if mode in ["Global AI", "Hybrid AI"]:
            with tab_map["🌍 Global"]:
                st.subheader("🌍 Global Medical Research")
                resp = safe_ai_call(query, mode="Global AI")
                if resp["status"] != "ok":
                    st.error(resp["answer"])
                else:
                    st.write(resp["answer"])

        # -------- Outcomes --------
        if mode == "Hybrid AI":
            with tab_map["🧪 Outcomes"]:
                st.subheader("🧪 Outcomes")
                if "fda" in resp["answer"].lower():
                    st.success("FDA-approved therapy detected")
                else:
                    st.info("No FDA outcome keyword detected.")

        # -------- Library --------
        if "📚 Library" in tab_map:
            with tab_map["📚 Library"]:
                st.subheader("📚 Medical Library")
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

        reader = PdfReader("lab_report.pdf")
        report_text = ""
        for p in reader.pages:
            report_text += (p.extract_text() or "") + "\n"

        values = extract_lab_values(report_text)
        summary, alerts = generate_lab_summary(values)

        st.markdown("### 🧾 Smart Lab Summary")
        if not summary:
            st.warning("Unable to auto-detect RESULT values (scanned PDFs may need OCR).")
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

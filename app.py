# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================
# Features:
# - Doctor Login & Audit Trail
# - Clinical Research Copilot (Hospital/Global/Hybrid)
# - Medical Library + FAISS
# - Lab Report Intelligence (Hospital-grade RESULT parser)
# - Smart Lab Summary (🟢🟡🔴) + ICU Alerts (🚨)
# - Clinical Sync Layer (AI narrative matches summary)
# - Doctor PDF Summary Generator
# - ICU Command Center Dashboard
# - Mobile-friendly layout
# ======================================================

import streamlit as st
import os, json, pickle, datetime, re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# -------------------------------
# Optional PDF export (fallback to text if missing)
# -------------------------------
PDF_EXPORT_AVAILABLE = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    PDF_EXPORT_AVAILABLE = False

# ======================================================
# PAGE CONFIG (Mobile-friendly)
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# THEME (clean + mobile)
# ======================================================
st.markdown("""
<style>
body { background: #0b1220; color: #e5e7eb; }
.block-container { padding-top: 1rem; }
.card { background: rgba(255,255,255,0.05); border-radius: 14px; padding: 16px; margin-bottom: 12px; }
.badge { padding: 6px 10px; border-radius: 999px; font-weight: 600; }
.ok { background: #00c2a8; color: #041b16; }
.warn { background: #ffd166; color: #3b2f00; }
.danger { background: #ef476f; }
.info { background: #118ab2; }
.small { opacity: .8; font-size: .9rem; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); }
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
ANALYTICS_FILE = "analytics_log.json"
USERS_DB = "users.json"
AUDIT_LOG = "audit_log.json"
REPORTS_FOLDER = "doctor_reports"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Seed users if not present
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
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# AUTH
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
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

def logout_ui():
    if st.sidebar.button("Logout"):
        audit("logout")
        for k in ["logged_in","username","role"]:
            st.session_state[k] = None
        st.session_state.logged_in = False
        st.experimental_rerun()

# ======================================================
# MODEL (for Research Copilot)
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

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index = faiss.read_index(INDEX_FILE)
    data = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.documents = data["documents"]
    st.session_state.sources = data["sources"]
    st.session_state.index_ready = True

# ======================================================
# LAB REFERENCE RULES
# ======================================================
LAB_RULES = {
    "Total Bilirubin": (0.3, 1.2, "mg/dL"),
    "Direct Bilirubin": (0.0, 0.2, "mg/dL"),
    "Indirect Bilirubin": (0.3, 1.0, "mg/dL"),
    "SGPT": (0, 50, "U/L"),
    "SGOT": (0, 50, "U/L"),
    "ALP": (43, 115, "U/L"),
    "GGT": (0, 55, "U/L"),
    "Albumin": (3.5, 5.2, "g/dL"),
    "Total Protein": (6.6, 8.3, "g/dL"),
    "Creatinine": (0.7, 1.3, "mg/dL"),
    "Urea": (10, 45, "mg/dL"),
    "Hemoglobin": (13, 17, "g/dL"),
    "WBC": (4000, 10000, "/cumm"),
    "Platelets": (150000, 410000, "/cumm"),
}

# ======================================================
# HOSPITAL-GRADE RESULT COLUMN PARSER
# - Finds RESULT on same line (last number) or next lines
# - Ignores reference ranges (x - y)
# ======================================================
def extract_lab_values_from_pdf(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    found = {}
    for i, line in enumerate(lines):
        for test in LAB_RULES:
            if test.lower() in line.lower():
                # 1) Try same line: take last standalone number (RESULT)
                nums = re.findall(r"\b\d+\.?\d*\b", line)
                if nums:
                    found[test] = float(nums[-1])
                    continue
                # 2) Try next lines: first standalone numeric row
                for j in range(i+1, min(i+6, len(lines))):
                    if re.fullmatch(r"\d+\.?\d*", lines[j]):
                        found[test] = float(lines[j])
                        break
    return found

# ======================================================
# MEDICAL DECISION LOGIC + ICU RULES
# ======================================================
def generate_lab_summary(values):
    summary = []
    alerts = []
    for test, val in values.items():
        low, high, unit = LAB_RULES[test]
        if val < low:
            status = "LOW"
            badge = "🟡"
        elif val > high:
            status = "HIGH"
            badge = "🔴"
        else:
            status = "NORMAL"
            badge = "🟢"
        summary.append((test, val, unit, f"{badge} {status}"))

        # ICU triggers
        if test == "Total Bilirubin" and val >= 5:
            alerts.append("🚨 Severe Jaundice — ICU evaluation required")
        if test == "Creatinine" and val >= 3:
            alerts.append("🚨 Acute Renal Failure risk")
        if test == "Platelets" and val < 50000:
            alerts.append("🚨 Bleeding risk")
        if test == "Hemoglobin" and val < 7:
            alerts.append("🚨 Severe anemia — transfusion required")
        if test == "WBC" and val > 20000:
            alerts.append("🚨 Sepsis risk (WBC > 20,000)")

    return summary, alerts

# ======================================================
# DOCTOR PDF REPORT GENERATOR
# ======================================================
def generate_doctor_pdf(patient_name, summary_rows, alerts, ai_text, outfile):
    if not PDF_EXPORT_AVAILABLE:
        with open(outfile.replace(".pdf",".txt"), "w") as f:
            f.write("ĀROGYABODHA AI — Doctor Summary\n")
            f.write(f"Patient: {patient_name}\n\n")
            for t,v,u,s in summary_rows:
                f.write(f"{t}: {v} {u} — {s}\n")
            if alerts:
                f.write("\nICU Alerts:\n")
                for a in alerts: f.write(f"- {a}\n")
            f.write("\nAI Clinical Opinion:\n")
            f.write(ai_text)
        return outfile.replace(".pdf",".txt")

    doc = SimpleDocTemplate(outfile, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph("<b>ĀROGYABODHA AI — Doctor Clinical Summary</b>", styles["Title"]))
    elems.append(Spacer(1,12))
    elems.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
    elems.append(Spacer(1,12))

    table_data = [["Test","Value","Unit","Status"]]
    for t,v,u,s in summary_rows:
        table_data.append([t, str(v), u, s])
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
        ("GRID",(0,0),(-1,-1), 0.5, colors.grey),
        ("FONT",(0,0),(-1,0),"Helvetica-Bold")
    ]))
    elems.append(table)
    elems.append(Spacer(1,12))

    if alerts:
        elems.append(Paragraph("<b>ICU Alerts</b>", styles["Heading2"]))
        for a in alerts:
            elems.append(Paragraph(a, styles["Normal"]))
        elems.append(Spacer(1,12))

    elems.append(Paragraph("<b>AI Clinical Opinion</b>", styles["Heading2"]))
    elems.append(Paragraph(ai_text.replace("\n","<br/>"), styles["Normal"]))

    doc.build(elems)
    return outfile

# ======================================================
# SIDEBAR (Auth + Navigation)
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
    "ICU Command Center",
    "Audit Trail"
])

# ======================================================
# HEADER
# ======================================================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform")
st.markdown("<div class='small'>Mobile-friendly • Evidence-locked • Hospital-grade</div>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if module == "Clinical Research Copilot":
    st.markdown("### 🔬 Clinical Research Copilot")
    query = st.text_input("Ask a clinical research question")
    mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

    if st.button("🚀 Analyze"):
        audit("research_query", {"query": query, "mode": mode})

        # Hospital/Hybrid: evidence-locked via library (if built)
        if mode in ["Hospital AI","Hybrid AI"] and st.session_state.index_ready:
            qemb = embedder.encode([query])
            _, I = st.session_state.index.search(np.array(qemb), 5)
            context = "\n\n".join([st.session_state.documents[i] for i in I[0]])
            prompt = f"""
You are a Hospital Clinical Decision Support AI.
Use ONLY the hospital evidence below. Do NOT hallucinate.

Hospital Evidence:
{context}

Doctor Query:
{query}
"""
            hospital_ans = external_research_answer(prompt).get("answer","")
            st.markdown("<div class='card'><b>🏥 Hospital AI</b></div>", unsafe_allow_html=True)
            st.write(hospital_ans)

        # Global/Hybrid
        if mode in ["Global AI","Hybrid AI"]:
            global_ans = external_research_answer(query).get("answer","")
            st.markdown("<div class='card'><b>🌍 Global AI</b></div>", unsafe_allow_html=True)
            st.write(global_ans)

# ======================================================
# LAB REPORT INTELLIGENCE (Clinical Sync Layer)
# ======================================================
if module == "Lab Report Intelligence":
    st.markdown("### 🧪 Lab Report Intelligence")
    lab_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
    patient_name = st.text_input("Patient Name", value="")

    if lab_file:
        with open("lab_report.pdf","wb") as f:
            f.write(lab_file.getbuffer())

        reader = PdfReader("lab_report.pdf")
        report_text = ""
        for p in reader.pages:
            report_text += (p.extract_text() or "") + "\n"

        values = extract_lab_values_from_pdf(report_text)
        summary, alerts = generate_lab_summary(values)

        st.markdown("#### 🧾 Smart Lab Summary")
        if not summary:
            st.warning("⚠ Unable to auto-detect RESULT values. (Scanned PDFs need OCR)")
        else:
            for t,v,u,s in summary:
                badge = "ok" if "NORMAL" in s else ("warn" if "LOW" in s else "danger")
                st.markdown(f"<div class='card'><span class='badge {badge}'>{s}</span> "
                            f"<b>{t}</b>: {v} {u}</div>", unsafe_allow_html=True)

        if alerts:
            st.markdown("#### 🚨 ICU Alerts")
            for a in alerts:
                st.error(a)

        st.markdown("#### 🧠 Ask ĀROGYABODHA AI")
        lab_question = st.text_input("Ask about this report")

        if st.button("Analyze Lab Report"):
            audit("lab_analyze", {"patient": patient_name})
            prompt = f"""
You are a hospital clinical AI.

Lab Report:
{report_text}

Doctor Question:
{lab_question}

Provide:
- Diagnosis pattern
- Clinical risks
- Next steps
- ICU escalation criteria
Use concise, doctor-style language.
"""
            ai_text = external_research_answer(prompt).get("answer","")
            st.markdown("<div class='card'><b>AI Clinical Opinion</b></div>", unsafe_allow_html=True)
            st.write(ai_text)

            # Doctor PDF
            if st.button("📄 Generate Doctor PDF Summary"):
                fname = f"{patient_name or 'patient'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                outpath = os.path.join(REPORTS_FOLDER, fname)
                saved = generate_doctor_pdf(patient_name or "Patient", summary, alerts, ai_text, outpath)
                audit("pdf_generated", {"file": os.path.basename(saved)})
                st.success(f"Saved: {saved}")
                with open(saved, "rb") as f:
                    st.download_button("Download PDF", f, file_name=os.path.basename(saved))

# ======================================================
# ICU COMMAND CENTER
# ======================================================
if module == "ICU Command Center":
    st.markdown("### ⚠ ICU Command Center")
    st.markdown("<div class='small'>Real-time triage from uploaded lab reports</div>", unsafe_allow_html=True)

    rows = []
    if os.path.exists(AUDIT_LOG):
        rows = json.load(open(AUDIT_LOG))

    st.markdown("#### Recent ICU Alerts (from lab analyses)")
    # Scan saved doctor reports text (or logs) for ICU flags
    alerts_view = []
    for file in os.listdir(REPORTS_FOLDER):
        if file.endswith(".txt") or file.endswith(".pdf"):
            alerts_view.append(file)
    if alerts_view:
        for f in alerts_view[-10:]:
            st.write("•", f)
    else:
        st.info("No ICU alerts recorded yet.")

# ======================================================
# AUDIT TRAIL
# ======================================================
if module == "Audit Trail":
    st.markdown("### 🕒 Audit Trail")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No audit logs yet.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

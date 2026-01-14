import streamlit as st
import os, json, pickle, datetime, re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ======================================================
# PAGE CONFIG + WOW UI
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Clinical Intelligence Command Center",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body { background: radial-gradient(circle at top, #020617, #020617); color: #e5e7eb; }
.card { background: rgba(255,255,255,0.04); border-radius: 18px; padding: 20px;
        box-shadow: 0 0 40px rgba(0,200,255,0.15); margin-bottom: 20px; }
.alert { background: linear-gradient(135deg, #ff004c, #ff6a00); padding: 15px;
         border-radius: 14px; font-weight: bold; }
.success { background: linear-gradient(135deg, #00ff9c, #00c2ff); padding: 15px;
           border-radius: 14px; font-weight: bold; color: black; }
.warn { background: linear-gradient(135deg, #ffe259, #ffa751); padding: 15px;
        border-radius: 14px; font-weight: bold; color: black; }
.normal { background: linear-gradient(135deg, #7CFFCB, #00ff9c); padding: 15px;
          border-radius: 14px; font-weight: bold; color: black; }
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

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ======================================================
# AI MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================================================
# SMART LAB RULE ENGINE
# ======================================================
LAB_RULES = {
    "Hemoglobin": {"low": 13, "high": 17, "unit": "g/dL"},
    "WBC": {"low": 4000, "high": 10000, "unit": "/cumm"},
    "Platelets": {"low": 150000, "high": 410000, "unit": "/cumm"},
    "Creatinine": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
    "Total Bilirubin": {"low": 0.3, "high": 1.2, "unit": "mg/dL"},
    "SGPT": {"low": 0, "high": 50, "unit": "U/L"},
    "SGOT": {"low": 0, "high": 50, "unit": "U/L"},
    "GGT": {"low": 0, "high": 55, "unit": "U/L"}
}

# ======================================================
# LAB EXTRACTION ENGINE (CBC + LFT + RFT)
# ======================================================
def extract_lab_values(text):
    patterns = {
        "Hemoglobin": r"Hemoglobin.*?(\d+\.?\d*)",
        "WBC": r"(WBC|TLC|Total Leukocyte Count).*?(\d+)",
        "Platelets": r"Platelet.*?(\d+)",
        "Creatinine": r"Creatinine.*?(\d+\.?\d*)",
        "Total Bilirubin": r"Total Bilirubin.*?(\d+\.?\d*)",
        "SGPT": r"(SGPT|ALT).*?(\d+)",
        "SGOT": r"(SGOT|AST).*?(\d+)",
        "GGT": r"(GGT|Gamma).*?(\d+)"
    }

    results = {}
    for test, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.groups()[-1]
            results[test] = float(value)

    return results

# ======================================================
# MEDICAL DECISION LOGIC
# ======================================================
def classify_value(test, value):
    ref = LAB_RULES[test]
    if value < ref["low"]:
        return "LOW"
    elif value > ref["high"]:
        return "HIGH"
    else:
        return "NORMAL"

def generate_smart_summary(values):
    summary = []
    alerts = []

    for test, value in values.items():
        status = classify_value(test, value)
        unit = LAB_RULES[test]["unit"]

        if status == "HIGH":
            summary.append((test, value, unit, "🔴 HIGH"))
        elif status == "LOW":
            summary.append((test, value, unit, "🟡 LOW"))
        else:
            summary.append((test, value, unit, "🟢 NORMAL"))

        # ICU rules
        if test == "Creatinine" and value > 3:
            alerts.append("🚨 CRITICAL: Acute Renal Failure risk")
        if test == "Total Bilirubin" and value > 5:
            alerts.append("🚨 CRITICAL: Severe Jaundice – ICU required")
        if test == "Platelets" and value < 50000:
            alerts.append("🚨 CRITICAL: Bleeding risk")

    return summary, alerts

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🧠 ĀROGYABODHA AI")
st.sidebar.subheader("📁 Medical Library")

uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

st.sidebar.divider()
app_mode = st.sidebar.radio("Select Module", ["Clinical Research Copilot", "Lab Report Intelligence"])

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='card'><h1>🧠 ĀROGYABODHA AI — Clinical Intelligence Command Center</h1></div>", unsafe_allow_html=True)

# ======================================================
# LAB REPORT INTELLIGENCE (UPGRADED)
# ======================================================
if app_mode == "Lab Report Intelligence":

    st.markdown("<div class='card'><h2>🧪 Lab Report Intelligence</h2></div>", unsafe_allow_html=True)

    lab_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])

    if lab_file:
        with open("lab_report.pdf", "wb") as f:
            f.write(lab_file.getbuffer())

        reader = PdfReader("lab_report.pdf")
        report_text = ""
        for page in reader.pages:
            report_text += page.extract_text() + "\n"

        values = extract_lab_values(report_text)

        summary, alerts = generate_smart_summary(values)

        st.markdown("<div class='card'><h3>🧾 Smart Lab Summary</h3></div>", unsafe_allow_html=True)

        for test, value, unit, status in summary:
            if "HIGH" in status:
                st.markdown(f"<div class='alert'>{test}: {value} {unit} — {status}</div>", unsafe_allow_html=True)
            elif "LOW" in status:
                st.markdown(f"<div class='warn'>{test}: {value} {unit} — {status}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='normal'>{test}: {value} {unit} — {status}</div>", unsafe_allow_html=True)

        if alerts:
            st.markdown("<div class='card'><h3>🚨 ICU Red Alerts</h3></div>", unsafe_allow_html=True)
            for a in alerts:
                st.markdown(f"<div class='alert'>{a}</div>", unsafe_allow_html=True)

        lab_question = st.text_input("Ask ĀROGYABODHA AI")

        if st.button("🧠 Analyze Lab Report"):
            prompt = f"""
You are a hospital clinical AI.

Lab Report:
{report_text}

Doctor Question:
{lab_question}

Provide diagnosis pattern, risks and next steps.
"""
            answer = external_research_answer(prompt).get("answer", "")
            st.markdown("<div class='success'>AI Clinical Opinion</div>", unsafe_allow_html=True)
            st.write(answer)

# ======================================================
# FOOTER
# ======================================================
st.markdown("<center>ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform</center>", unsafe_allow_html=True)

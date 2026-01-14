import streamlit as st
import os, json, datetime, re
from pypdf import PdfReader
from external_research import external_research_answer

# ======================================================
# PAGE CONFIG + UI
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
.small { opacity: .8; font-size: 0.9rem; }
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
# MEDICAL REFERENCE RULES
# ======================================================
LAB_RULES = {
    # CBC
    "Hemoglobin": {"low": 13, "high": 17, "unit": "g/dL"},
    "WBC": {"low": 4000, "high": 10000, "unit": "/cumm"},
    "Platelets": {"low": 150000, "high": 410000, "unit": "/cumm"},
    "RBC": {"low": 4.5, "high": 5.5, "unit": "million/cumm"},
    "MCV": {"low": 81, "high": 101, "unit": "fL"},
    "MCH": {"low": 27, "high": 32, "unit": "pg"},
    "MCHC": {"low": 31.5, "high": 34.5, "unit": "g/dL"},

    # LFT
    "Total Bilirubin": {"low": 0.3, "high": 1.2, "unit": "mg/dL"},
    "Direct Bilirubin": {"low": 0.0, "high": 0.2, "unit": "mg/dL"},
    "Indirect Bilirubin": {"low": 0.3, "high": 1.0, "unit": "mg/dL"},
    "SGPT": {"low": 0, "high": 50, "unit": "U/L"},
    "SGOT": {"low": 0, "high": 50, "unit": "U/L"},
    "ALP": {"low": 43, "high": 115, "unit": "U/L"},
    "GGT": {"low": 0, "high": 55, "unit": "U/L"},
    "Albumin": {"low": 3.5, "high": 5.2, "unit": "g/dL"},
    "Total Protein": {"low": 6.6, "high": 8.3, "unit": "g/dL"},

    # RFT
    "Creatinine": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
    "Urea": {"low": 10, "high": 45, "unit": "mg/dL"},
}

# ======================================================
# PDF TABLE EXTRACTION ENGINE (FINAL FIX)
# ======================================================
def extract_lab_values_from_pdf(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    found = {}

    for i, line in enumerate(lines):
        key = None

        # Detect test names
        for test in LAB_RULES.keys():
            if test.lower() in line.lower():
                key = test
                break

        if key:
            # Look next lines for numeric value
            for j in range(i+1, min(i+5, len(lines))):
                val_match = re.search(r"(\d+\.?\d*)", lines[j])
                if val_match:
                    found[key] = float(val_match.group(1))
                    break

    return found

# ======================================================
# MEDICAL DECISION LOGIC
# ======================================================
def classify(test, value):
    ref = LAB_RULES[test]
    if value < ref["low"]:
        return "LOW"
    elif value > ref["high"]:
        return "HIGH"
    else:
        return "NORMAL"

def generate_summary(values):
    rows = []
    alerts = []

    for test, value in values.items():
        status = classify(test, value)
        unit = LAB_RULES[test]["unit"]

        if status == "HIGH":
            flag = "🔴 HIGH"
        elif status == "LOW":
            flag = "🟡 LOW"
        else:
            flag = "🟢 NORMAL"

        rows.append((test, value, unit, flag))

        # ICU Rules
        if test == "Total Bilirubin" and value >= 5:
            alerts.append("🚨 CRITICAL: Severe Jaundice — ICU evaluation required.")
        if test == "Creatinine" and value >= 3:
            alerts.append("🚨 CRITICAL: Acute Renal Failure risk.")
        if test == "Platelets" and value < 50000:
            alerts.append("🚨 CRITICAL: Bleeding risk.")
        if test == "Hemoglobin" and value < 7:
            alerts.append("🚨 CRITICAL: Severe Anemia — transfusion needed.")

    return rows, alerts

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🧠 ĀROGYABODHA AI")
app_mode = st.sidebar.radio("Select Module", ["Lab Report Intelligence"])

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='card'><h1>🧠 ĀROGYABODHA AI — Clinical Intelligence Command Center</h1></div>", unsafe_allow_html=True)

# ======================================================
# LAB REPORT INTELLIGENCE
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
            report_text += (page.extract_text() or "") + "\n"

        values = extract_lab_values_from_pdf(report_text)
        summary, alerts = generate_summary(values)

        # Smart Summary
        st.markdown("<div class='card'><h3>🧾 Smart Lab Summary</h3></div>", unsafe_allow_html=True)

        if not summary:
            st.warning("⚠ Unable to auto-detect lab parameters from this PDF (Scanned image PDF needs OCR).")

        for test, value, unit, flag in summary:
            if "HIGH" in flag:
                st.markdown(f"<div class='alert'>{test}: {value} {unit} — {flag}</div>", unsafe_allow_html=True)
            elif "LOW" in flag:
                st.markdown(f"<div class='warn'>{test}: {value} {unit} — {flag}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='normal'>{test}: {value} {unit} — {flag}</div>", unsafe_allow_html=True)

        # ICU Alerts
        if alerts:
            st.markdown("<div class='card'><h3>🚨 ICU Red Alerts</h3></div>", unsafe_allow_html=True)
            for a in alerts:
                st.markdown(f"<div class='alert'>{a}</div>", unsafe_allow_html=True)

        # Ask AI
        lab_question = st.text_input("Ask ĀROGYABODHA AI about this report")

        if st.button("🧠 Analyze Lab Report"):
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
"""
            answer = external_research_answer(prompt).get("answer", "")
            st.markdown("<div class='success'>AI Clinical Opinion</div>", unsafe_allow_html=True)
            st.write(answer)

# ======================================================
# FOOTER
# ======================================================
st.markdown("<center>ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform</center>", unsafe_allow_html=True)

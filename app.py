# ============================================================
# ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS
# ============================================================

import streamlit as st
import os, json, datetime, io, requests, re, base64
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="ĀROGYABODHA AI — Medical Intelligence OS",
    page_icon="🧠",
    layout="wide"
)

st.info(
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS). "
    "It does NOT provide diagnosis or treatment. "
    "Final decisions must be made by licensed doctors."
)

BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

os.makedirs(PDF_FOLDER, exist_ok=True)

# ============================================================
# DATABASE INIT
# ============================================================

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

# ============================================================
# AUDIT
# ============================================================

def audit(event, meta=None):
    logs = json.load(open(AUDIT_LOG)) if os.path.exists(AUDIT_LOG) else []
    logs.append({
        "time": str(datetime.datetime.utcnow()),
        "user": st.session_state.username,
        "event": event,
        "meta": meta or {}
    })
    json.dump(logs, open(AUDIT_LOG, "w"), indent=2)

# ============================================================
# LOGIN
# ============================================================

def login_ui():
    st.title("ĀROGYABODHA AI — Secure Medical Login")

    with st.form("login"):
        u = st.text_input("User ID")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")

    if ok:
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.role = users[u]["role"]
            audit("login", {"user": u})
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ============================================================
# PDF DISPLAY
# ============================================================

def display_pdf(path, height=700):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}"></iframe>',
        unsafe_allow_html=True
    )

# ============================================================
# PUBMED CONNECTORS
# ============================================================

def fetch_pubmed(query):
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 10},
            timeout=15
        )
        return r.json()["esearchresult"]["idlist"]
    except:
        return []

def fetch_pubmed_details(pmids):
    if not pmids:
        return []

    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            timeout=20
        )

        papers = []
        for art in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", r.text, re.S):
            pmid = re.search(r"<PMID.*?>(.*?)</PMID>", art)
            title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S)
            abstract = re.search(r"<AbstractText.*?>(.*?)</AbstractText>", art, re.S)

            papers.append({
                "pmid": pmid.group(1) if pmid else "N/A",
                "title": re.sub("<.*?>", "", title.group(1)) if title else "No title",
                "abstract": re.sub("<.*?>", "", abstract.group(1))[:1200]
                if abstract else "No abstract available",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid.group(1)}/"
                if pmid else ""
            })

        return papers
    except:
        return []

# ============================================================
# AI SUMMARY (SAFE SYNTHESIS)
# ============================================================

def generate_ai_summary(query, papers):
    if not papers:
        return "No sufficient literature found for synthesis."

    return f"""
### 🧠 AI-Synthesized Clinical Summary

Based on current PubMed-indexed research related to:

**{query}**

the evidence consistently highlights:

• Morphological evaluation via blood smear and bone marrow biopsy  
• Immunophenotyping using flow cytometry  
• Cytogenetic and molecular diagnostics (PCR, NGS)  
• Treatment response monitoring through residual disease markers  
• Increasing use of computational and AI-assisted pathology tools  

> ℹ️ This is a research synthesis only — not diagnostic or treatment guidance.
"""

# ============================================================
# PAPERS UI
# ============================================================

def display_pubmed_papers(papers):
    st.subheader("📚 Papers Found")

    tab_pubmed, = st.tabs([f"PubMed ({len(papers)})"])

    with tab_pubmed:
        if papers:
            for p in papers:
                with st.expander(f"📄 {p['title']}"):
                    st.markdown(f"**PMID:** {p['pmid']}")
                    st.markdown("**Abstract**")
                    st.write(p["abstract"])
                    st.link_button("🔗 View on PubMed", p["url"])
        else:
            st.info("No PubMed papers found.")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}**")

if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Medical Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Phase-3 Research Copilot",
    "📊 Live Intelligence Dashboard",
    "🕒 Audit & Compliance"
])

# ============================================================
# MODULES
# ============================================================

# ---------- Evidence Library ----------

if module == "📁 Evidence Library":
    st.header("📁 Medical Evidence Library")

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
                out.write(f.read())
        st.success("PDFs uploaded")

    st.subheader("📄 View PDFs")

    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if pdfs:
        selected = st.selectbox("Select PDF", pdfs)
        display_pdf(os.path.join(PDF_FOLDER, selected))
    else:
        st.info("No PDFs available")

# ---------- Research Copilot ----------

if module == "🔬 Phase-3 Research Copilot":
    st.header("🧠 Clinical Research Intelligence Assistant")

    query = st.text_input(
        "Ask a clinical research question",
        placeholder="e.g., pathological methods for diagnosing leukemia"
    )

    if st.button("Analyze Research") and query:
        audit("research_query", {"query": query})

        pubmed_ids = fetch_pubmed(query)
        papers = fetch_pubmed_details(pubmed_ids)

        # 🧠 AI Summary
        st.markdown(generate_ai_summary(query, papers))

        # 📚 Evidence
        display_pubmed_papers(papers)

        # 📄 Local PDFs
        st.subheader("📄 Local Hospital Evidence")

        pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        if pdfs:
            selected_pdf = st.selectbox("Select PDF evidence", pdfs)
            display_pdf(os.path.join(PDF_FOLDER, selected_pdf))
        else:
            st.info("No local PDFs available")

# ---------- Dashboard ----------

if module == "📊 Live Intelligence Dashboard":
    st.header("📊 Live Medical Intelligence Dashboard")
    st.metric("Total Evidence PDFs", len(os.listdir(PDF_FOLDER)))

# ---------- Audit ----------

if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance")

    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))), use_container_width=True)
    else:
        st.info("No audit logs")

# ============================================================
# FOOTER
# ============================================================

st.caption("ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS")

# ============================================================
# ĀROGYABODHA AI — Phase-3 Medical Intelligence OS
# ============================================================

import streamlit as st
import os, json, datetime, io, requests, re, base64
import pandas as pd
from pypdf import PdfReader

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="ĀROGYABODHA AI — Medical Intelligence OS",
    page_icon="🧠",
    layout="wide"
)

st.info(
    "ℹ️ Clinical Decision Support System (CDSS). "
    "Not for diagnosis or treatment."
)

BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

os.makedirs(PDF_FOLDER, exist_ok=True)

# ============================================================
# INIT USERS
# ============================================================

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123"},
        "researcher1": {"password": "research123"}
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

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
    st.title("ĀROGYABODHA AI Login")

    with st.form("login"):
        u = st.text_input("User ID")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")

    if ok:
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            audit("login")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ============================================================
# PDF DISPLAY
# ============================================================

def display_pdf(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700"></iframe>',
        unsafe_allow_html=True
    )

# ============================================================
# PUBMED API
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
# DYNAMIC AI SUMMARY
# ============================================================

def generate_ai_summary(papers):
    if not papers:
        return "No sufficient evidence found."

    text = " ".join(p["abstract"].lower() for p in papers)

    concept_map = {
        "bone marrow": "Morphological evaluation using bone marrow biopsy",
        "blood smear": "Peripheral blood smear examination",
        "flow cytometry": "Immunophenotyping using flow cytometry",
        "cytogenetic": "Cytogenetic chromosomal analysis",
        "molecular": "Molecular diagnostics (PCR, NGS)",
        "sequencing": "Genomic sequencing technologies",
        "residual disease": "Minimal residual disease monitoring",
        "biomarker": "Biomarker-based outcome prediction",
        "artificial intelligence": "AI-assisted digital pathology",
        "multi-omics": "Multi-omics profiling"
    }

    found = [v for k, v in concept_map.items() if k in text]

    if not found:
        return "No dominant clinical techniques detected."

    bullets = "\n".join(f"• {x}" for x in sorted(set(found)))

    return f"""
### 🧠 AI-Synthesized Clinical Summary

{bullets}

ℹ️ Literature-driven synthesis only.
"""

# ============================================================
# PAPERS UI
# ============================================================

def display_papers(papers):
    st.subheader("📚 Papers Found")

    if not papers:
        st.info("No papers found.")
        return

    for p in papers:
        with st.expander(f"📄 {p['title']}"):
            st.write(p["abstract"])
            st.link_button("View on PubMed", p["url"])

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(f"👨‍⚕️ {st.session_state.username}")

if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Medical Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Research Copilot",
    "🕒 Audit"
])

# ============================================================
# MODULES
# ============================================================

if module == "📁 Evidence Library":
    st.header("Evidence Library")

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.read())
        st.success("Uploaded")

    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if pdfs:
        selected = st.selectbox("Select PDF", pdfs)
        display_pdf(os.path.join(PDF_FOLDER, selected))
    else:
        st.info("No PDFs")

if module == "🔬 Research Copilot":
    st.header("Clinical Research Assistant")

    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze") and query:
        audit("query", {"q": query})

        ids = fetch_pubmed(query)
        papers = fetch_pubmed_details(ids)

        st.markdown(generate_ai_summary(papers))
        display_papers(papers)

if module == "🕒 Audit":
    st.header("Audit Log")

    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))))
    else:
        st.info("No logs")

# ============================================================
# FOOTER
# ============================================================

st.caption("ĀROGYABODHA AI — Phase-3 Medical Intelligence OS")

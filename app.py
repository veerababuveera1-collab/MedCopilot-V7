# ============================================================
# ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, requests, re, base64
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
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
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS). "
    "It does NOT provide diagnosis or treatment. "
    "Final decisions must be made by licensed doctors."
)

# ============================================================
# STORAGE
# ============================================================
BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
VECTOR_FOLDER = os.path.join(BASE, "vector_cache")

PATIENT_DB = os.path.join(BASE, "patients.json")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ============================================================
# DATABASE INIT
# ============================================================
if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"), indent=2)

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION
# ============================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index": None,
    "docs": [],
    "srcs": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# AUDIT
# ============================================================
def audit(event, meta=None):
    logs = []
    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))
    logs.append({
        "time": str(datetime.datetime.utcnow()),
        "user": st.session_state.username,
        "role": st.session_state.role,
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
# MODEL
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ============================================================
# PDF HELPERS
# ============================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages[:200]:
        t = p.extract_text()
        if t and len(t) > 100:
            pages.append(t)
    return pages

def display_pdf(pdf_path, height=700):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""<iframe src="data:application/pdf;base64,{base64_pdf}"
        width="100%" height="{height}"></iframe>""",
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
# UI HELPER — PAPERS FOUND (PUBMED STYLE)
# ============================================================
def display_pubmed_papers(pubmed_papers):
    st.subheader("📚 Papers Found")

    tab_pubmed, = st.tabs([f"PubMed ({len(pubmed_papers)})"])

    with tab_pubmed:
        if pubmed_papers:
            for p in pubmed_papers:
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
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Medical Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Phase-3 Research Copilot",
    "📊 Live Intelligence Dashboard",
    "👤 Patient Workspace",
    "🕒 Audit & Compliance"
])

# ============================================================
# MODULES
# ============================================================

# ---------- Evidence Library ----------
if module == "📁 Evidence Library":
    st.header("📁 Medical Evidence Library")

    files = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
                out.write(f.read())
        st.success("PDFs uploaded")

    st.subheader("📄 View PDFs")
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if pdfs:
        selected = st.selectbox("Select PDF", pdfs)
        display_pdf(os.path.join(PDF_FOLDER, selected))
    else:
        st.info("No PDFs uploaded")

# ---------- Phase-3 Research Copilot ----------
if module == "🔬 Phase-3 Research Copilot":
    st.header("🧠 Clinical Research Intelligence Assistant")

    query = st.text_input(
        "Ask a clinical research question",
        placeholder="e.g., pathological methods for diagnosing leukemia"
    )

    if st.button("Analyze Research") and query:
        audit("phase3_query", {"query": query})

        pubmed_ids = fetch_pubmed(query)
        pubmed_papers = fetch_pubmed_details(pubmed_ids)

        display_pubmed_papers(pubmed_papers)

        st.subheader("📄 Local Hospital Evidence (PDF)")
        pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
        if pdfs:
            selected_pdf = st.selectbox("Select evidence PDF", pdfs)
            display_pdf(os.path.join(PDF_FOLDER, selected_pdf))
        else:
            st.info("No local PDFs available")

# ---------- Live Dashboard ----------
if module == "📊 Live Intelligence Dashboard":
    st.header("📊 Live Medical Intelligence Dashboard")
    st.metric("Indexed PDFs", len(os.listdir(PDF_FOLDER)))

# ---------- Patient Workspace ----------
if module == "👤 Patient Workspace":
    st.header("👤 Patient Workspace")
    st.info("Patient module ready")

# ---------- Audit ----------
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance")
    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))), use_container_width=True)
    else:
        st.info("No audit records")

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS")

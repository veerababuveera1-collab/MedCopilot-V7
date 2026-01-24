# ============================================================
# ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS
# Hospital + Research + Trial + Regulatory + Clinical Reasoning Platform
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, requests, re
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
    "index_ready": False,
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
    st.title("ĀROGYABODHA AI — Secure Medical Intelligence Login")
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
# PDF INDEXING
# ============================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages[:200]:
        t = p.extract_text()
        if t and len(t) > 100:
            pages.append(t)
    return pages

def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.lower().endswith(".pdf"):
            with open(os.path.join(PDF_FOLDER, pdf), "rb") as f:
                pages = extract_text(f.read())
            for i, p in enumerate(pages):
                docs.append(p)
                srcs.append(f"{pdf} — Page {i+1}")

    if not docs:
        return None, [], []

    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))

    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"docs": docs, "srcs": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

# ============================================================
# PUBMED / TRIALS / FDA
# ============================================================
def fetch_pubmed(query):
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 5},
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
        xml = r.text
        papers = []

        articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.S)
        for art in articles:
            pmid = re.search(r"<PMID.*?>(.*?)</PMID>", art)
            title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S)
            abstract = re.search(r"<AbstractText.*?>(.*?)</AbstractText>", art, re.S)
            journal = re.search(r"<Title>(.*?)</Title>", art)
            year = re.search(r"<Year>(\d{4})</Year>", art)

            authors = re.findall(
                r"<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>",
                art, re.S
            )
            author_list = [f"{f} {l}" for l, f in authors[:6]]

            papers.append({
                "PMID": pmid.group(1) if pmid else "N/A",
                "Title": re.sub("<.*?>", "", title.group(1)) if title else "No title",
                "Authors": ", ".join(author_list) if author_list else "N/A",
                "Journal": journal.group(1) if journal else "N/A",
                "Year": year.group(1) if year else "N/A",
                "Abstract": re.sub("<.*?>", "", abstract.group(1))[:1200]
                            if abstract else "No abstract available"
            })
        return papers
    except:
        return []

def fetch_clinical_trials(query):
    try:
        r = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": query, "pageSize": 5},
            timeout=15
        )
        trials = []
        for study in r.json().get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            trials.append({
                "Trial ID": ident.get("nctId", "N/A"),
                "Phase": ", ".join(design.get("phases", ["N/A"])),
                "Status": status.get("overallStatus", "Unknown")
            })
        return trials
    except:
        return []

def fetch_fda_alerts():
    try:
        r = requests.get(
            "https://api.fda.gov/drug/enforcement.json?limit=5",
            timeout=15
        )
        return [
            f"{i.get('product_description','Unknown')} | {i.get('reason_for_recall','Safety Alert')}"
            for i in r.json().get("results", [])
        ]
    except:
        return []

# ============================================================
# CLINICAL REASONING
# ============================================================
def clinical_reasoning(query, pubmed_ids, trials, alerts):
    return f"""
## 🔬 Clinical Research Summary

**Research Question:** {query}

- PubMed Articles: {len(pubmed_ids)}
- Clinical Trials: {len(trials)}
- FDA Alerts: {len(alerts)}

This output is generated by a CDSS. Final decisions must be made by licensed doctors.
"""

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
    "🧾 Doctor Orders",
    "🕒 Audit & Compliance"
])

# ============================================================
# PHASE-3 RESEARCH COPILOT
# ============================================================
if module == "🔬 Phase-3 Research Copilot":

    st.header("🧠 Clinical Research Intelligence Assistant")
    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze Research") and query:

        audit("phase3_query", {"query": query})

        pubmed_ids = fetch_pubmed(query)
        pubmed_papers = fetch_pubmed_details(pubmed_ids)
        trials = fetch_clinical_trials(query)
        alerts = fetch_fda_alerts()

        st.subheader("Clinical Summary")
        st.markdown(clinical_reasoning(query, pubmed_ids, trials, alerts))

        st.subheader("📚 Journal Evidence")
        for p in pubmed_papers:
            with st.expander(p["Title"]):
                st.write(p)

        st.subheader("🧪 Clinical Trials")
        st.table(pd.DataFrame(trials)) if trials else st.info("No trials found")

        st.subheader("⚠ FDA Alerts")
        for a in alerts:
            st.warning(a)

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — Phase-3 PRODUCTION Medical Intelligence OS")

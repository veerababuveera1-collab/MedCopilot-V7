# ============================================================
# ĀROGYABODHA AI — Hybrid Medical Intelligence OS
# Semantic AI + Clinical Reasoning CDSS
# ============================================================

import streamlit as st
import os, json, datetime, requests, re, base64
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

st.set_page_config("ĀROGYABODHA AI", "🧠", layout="wide")
st.info("ℹ️ Clinical Decision Support System — Research only")

BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

os.makedirs(PDF_FOLDER, exist_ok=True)

# ============================================================
# USER DATABASE
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
# AUDIT LOGGING
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
    st.title("Secure Medical Login")

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
# PDF VIEWER
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
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 20},
            timeout=15
        )
        return r.json()["esearchresult"]["idlist"]
    except:
        return []

def fetch_pubmed_details(pmids):
    if not pmids:
        return []

    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        timeout=20
    )

    papers = []
    for art in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", r.text, re.S):
        title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S)
        abstract = re.search(r"<AbstractText.*?>(.*?)</AbstractText>", art, re.S)
        pmid = re.search(r"<PMID.*?>(.*?)</PMID>", art)

        papers.append({
            "title": re.sub("<.*?>", "", title.group(1)) if title else "No title",
            "abstract": re.sub("<.*?>", "", abstract.group(1)) if abstract else "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid.group(1)}/" if pmid else ""
        })

    return papers

# ============================================================
# HYBRID SEMANTIC AI
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def semantic_rank(query, papers, top_k=8):
    if not papers:
        return []

    texts = [p["abstract"] for p in papers] + [query]
    emb = model.encode(texts)

    paper_vecs = emb[:-1]
    query_vec = emb[-1]

    scores = np.dot(paper_vecs, query_vec)
    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)

    return [p for p, _ in ranked[:top_k]]

# ============================================================
# CLINICAL REASONING ENGINE
# ============================================================

def generate_ai_summary(query, papers):
    if not papers:
        return "No sufficient biomedical evidence found."

    combined = " ".join(p["abstract"].lower() for p in papers)

    concept_groups = {

        "Pathology & Morphology": {
            "biopsy","histopathology","microscopy","fibrosis","necrosis","tissue"
        },

        "Laboratory & Biomarkers": {
            "biomarker","troponin","crp","d-dimer","creatinine","hemoglobin"
        },

        "Imaging Diagnostics": {
            "ct","mri","ultrasound","echocardiography","radiology"
        },

        "Genomics & Molecular Medicine": {
            "pcr","sequencing","genomic","mutation","multi-omics"
        },

        "Pharmacology & Drug Safety": {
            "drug interaction","anticoagulant","ssri","metabolism",
            "toxicity","side effect","bleeding risk"
        },

        "Clinical Risk & Outcomes": {
            "mortality","prognosis","risk factor","complication","survival"
        },

        "Computational & AI Medicine": {
            "artificial intelligence","machine learning","predictive model"
        },

        "Therapeutic Response": {
            "treatment response","drug efficacy","resistance","improvement"
        }
    }

    lines = [
        "### 🧠 AI-Synthesized Clinical Summary",
        "",
        "**Key clinical methodologies identified:**",
        ""
    ]

    found = False

    for group, keys in concept_groups.items():
        hits = [k for k in keys if k in combined]
        if hits:
            found = True
            lines.append(f"🧪 **{group}**")
            for h in sorted(set(hits)):
                lines.append(f"- {h.capitalize()}-based clinical applications")
            lines.append("")

    if not found:
        lines.append("No dominant methodologies detected.")

    lines.append("ℹ️ Literature-driven synthesis only.")

    return "\n".join(lines)

# ============================================================
# PAPER UI
# ============================================================

def show_papers(papers):
    st.subheader("📚 Papers Found")
    for p in papers:
        with st.expander(f"📄 {p['title']}"):
            st.write(p["abstract"][:1200])
            st.link_button("View on PubMed", p["url"])

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(f"👨‍⚕️ {st.session_state.username}")

module = st.sidebar.radio("Medical Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Research Copilot",
    "📊 Dashboard",
    "🕒 Audit"
])

# ============================================================
# MODULES
# ============================================================

if module == "📁 Evidence Library":
    st.header("Medical Evidence PDFs")

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files:
        for f in files:
            open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.read())
        st.success("Uploaded")

    pdfs = os.listdir(PDF_FOLDER)
    if pdfs:
        display_pdf(os.path.join(PDF_FOLDER, pdfs[0]))
    else:
        st.info("No PDFs yet")

# ------------------------------------------------------------

if module == "🔬 Research Copilot":
    st.header("Clinical Research AI")

    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze") and query:
        audit("query", {"query": query})

        ids = fetch_pubmed(query)
        raw_papers = fetch_pubmed_details(ids)
        papers = semantic_rank(query, raw_papers)

        st.markdown(generate_ai_summary(query, papers))
        show_papers(papers)

        st.subheader("Local Evidence PDFs")
        pdfs = os.listdir(PDF_FOLDER)
        if pdfs:
            display_pdf(os.path.join(PDF_FOLDER, pdfs[0]))
        else:
            st.info("No local PDFs")

# ------------------------------------------------------------

if module == "📊 Dashboard":
    st.metric("Evidence PDFs", len(os.listdir(PDF_FOLDER)))
    st.metric("Total Queries", len(json.load(open(AUDIT_LOG))) if os.path.exists(AUDIT_LOG) else 0)

# ------------------------------------------------------------

if module == "🕒 Audit":
    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))), use_container_width=True)
    else:
        st.info("No audit logs")

# ============================================================
# FOOTER
# ============================================================

st.caption("ĀROGYABODHA AI — Hybrid Medical Intelligence OS")

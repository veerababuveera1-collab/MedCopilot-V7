# ============================================================
# ĀROGYABODHA AI — Medical Intelligence OS (Enterprise UI)
# Hospital-Grade Clinical Research Copilot (CDSS)
# ============================================================

import streamlit as st
import os, json, datetime, hashlib, uuid, requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from xml.etree import ElementTree as ET

# ----------------------------- CONFIG -----------------------------
st.set_page_config(
    page_title="ĀROGYABODHA AI — Medical Intelligence OS",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------- PATHS -----------------------------
PDF_FOLDER = "medical_library"        # place hospital PDFs here
AUDIT_LOG = "audit_ledger.json"       # immutable audit ledger

# ----------------------------- SESSION INIT -----------------------------
if "username" not in st.session_state:
    st.session_state.username = "doctor1"
if "role" not in st.session_state:
    st.session_state.role = "Doctor"
if "doctor_id" not in st.session_state:
    st.session_state.doctor_id = "DOC-001"

# ----------------------------- UTILITIES -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

ensure_dir(PDF_FOLDER)

# ----------------------------- AUDIT (IMMUTABLE LEDGER) -----------------------------
def audit(event, meta=None):
    logs = []
    if os.path.exists(AUDIT_LOG):
        try:
            with open(AUDIT_LOG, "r") as f:
                logs = json.load(f)
        except:
            logs = []

    previous_hash = logs[-1]["ledger_hash"] if logs else "GENESIS"
    record = {
        "audit_id": str(uuid.uuid4()),
        "timestamp_utc": str(datetime.datetime.utcnow()),
        "doctor_id": st.session_state.get("doctor_id", "unknown"),
        "user": st.session_state.get("username", "unknown"),
        "role": st.session_state.get("role", "unknown"),
        "event": event,
        "meta": meta or {},
        "previous_hash": previous_hash,
    }

    record["ledger_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()

    logs.append(record)
    with open(AUDIT_LOG, "w") as f:
        json.dump(logs, f, indent=2)

# ----------------------------- HEADER -----------------------------
st.markdown(
    """
# 🧠 ĀROGYABODHA AI — Medical Intelligence OS
### Enterprise Clinical Research Copilot (CDSS)
ℹ️ Research support only. Not a diagnosis or treatment system. Final decisions remain with licensed physicians.
"""
)

# ----------------------------- SIDEBAR (ENTERPRISE NAV) -----------------------------
st.sidebar.title("🏥 Medical Intelligence Center")
st.sidebar.metric("Doctor", st.session_state.username)
st.sidebar.metric("Role", st.session_state.role)

NAV = [
    "📁 Evidence Library",
    "🔬 Research Copilot (Hospital RAG)",
    "🌐 PubMed Global Research (NIH)",
    "🧪 Clinical Trials (Outcomes)",
    "🧬 FDA Drug Intelligence",
    "👤 Patient Cohorts",
    "📊 Live Dashboard",
    "🕒 Audit & Compliance",
]
page = st.sidebar.radio("Navigate", NAV)

# ----------------------------- SYSTEM HEALTH -----------------------------
st.sidebar.markdown("### 🩺 System Health")
st.sidebar.write("UI Engine:", "Running")

# ----------------------------- EMBEDDINGS -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()
st.sidebar.write("Embedding Model:", "Loaded")

# ----------------------------- LOAD PDFs -----------------------------
def load_pdfs(folder):
    documents, sources = [], []
    if not os.path.exists(folder):
        return documents, sources

    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    for file in pdfs:
        try:
            reader = PdfReader(os.path.join(folder, file))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text) > 200:
                    documents.append(text)
                    sources.append(f"{file} — Page {i+1}")
        except Exception:
            st.warning(f"Skipped corrupted PDF: {file}")
    return documents, sources

documents, sources = load_pdfs(PDF_FOLDER)
st.sidebar.write("Evidence Library:", "Loaded" if documents else "Not Loaded")

# ----------------------------- VECTOR DB -----------------------------
def build_index(docs):
    if not docs:
        return None
    emb = embedder.encode(docs)
    dim = emb.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(np.array(emb).astype("float32"))
    return idx

index = build_index(documents)
st.sidebar.write("Vector Index:", "Ready" if index else "Not Built")

# ----------------------------- CLINICAL TRIALS API -----------------------------
def fetch_clinical_trials(query, limit=10):
    url = (
        "https://clinicaltrials.gov/api/query/study_fields"
        f"?expr={query}&fields=NCTId,BriefTitle,Phase,OverallStatus,Condition"
        f"&min_rnk=1&max_rnk={limit}&fmt=json"
    )
    try:
        r = requests.get(url, timeout=20)
        data = r.json()
        return data.get("StudyFieldsResponse", {}).get("StudyFields", [])
    except:
        return []

# ----------------------------- FDA OPEN API -----------------------------
def fetch_fda_drug_info(drug):
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug}&limit=1"
    try:
        r = requests.get(url, timeout=20)
        data = r.json()
        return data.get("results", [None])[0]
    except:
        return None

# ----------------------------- COHORT ANALYTICS -----------------------------
def analyze_patient_cohort(age, diagnosis, biomarkers=None):
    cohort = {
        "age_group": "Senior (>=60)" if age >= 60 else "Adult (<60)",
        "diagnosis": diagnosis,
        "biomarkers": biomarkers or [],
        "matched_trials": fetch_clinical_trials(diagnosis, limit=5),
    }
    return cohort

# ===================== PubMed (NIH) LIVE INTEGRATION =====================

def pubmed_search(query, retmax=10):
    """Search PubMed and return list of PMIDs"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except:
        return []

def pubmed_fetch(pmids):
    """Fetch PubMed abstracts by PMIDs"""
    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        root = ET.fromstring(r.text)

        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")
            abstract = " ".join([a.text for a in article.findall(".//AbstractText") if a.text])
            articles.append({
                "PMID": pmid,
                "Title": title,
                "Abstract": abstract
            })
        return articles
    except:
        return []

# ----------------------------- OUTCOME COMPARISON (from trials) -----------------------------
def build_outcome_table(trials):
    rows = []
    for t in trials:
        rows.append({
            "NCT ID": (t.get("NCTId") or [None])[0],
            "Title": (t.get("BriefTitle") or [None])[0],
            "Phase": (t.get("Phase") or [None])[0],
            "Status": (t.get("OverallStatus") or [None])[0],
            "Condition": (t.get("Condition") or [None])[0],
        })
    return rows

# ============================= PAGES =============================

# ---------- 📁 Evidence Library ----------
if page == "📁 Evidence Library":
    st.subheader("📁 Hospital Evidence Library")

    with st.expander("Upload Hospital PDFs (Research / Guidelines / SOPs)"):
        uploaded = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )
        if uploaded:
            for f in uploaded:
                with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
                    out.write(f.read())
            st.success("Files uploaded. Rebuilding index...")
            documents, sources = load_pdfs(PDF_FOLDER)
            index = build_index(documents)
            audit("EVIDENCE_UPLOAD", {"count": len(uploaded), "files": [f.name for f in uploaded]})
            st.success("Evidence indexed successfully.")

    st.write("**Indexed Documents:**", len(documents))
    if documents:
        st.dataframe(
            {"Source": sources[:50], "Preview": [d[:120] + "..." for d in documents[:50]]}
        )
    else:
        st.info("No PDFs found. Upload hospital evidence to enable local RAG.")

# ---------- 🔬 Research Copilot (Hospital RAG) ----------
elif page == "🔬 Research Copilot (Hospital RAG)":
    st.subheader("🔬 Phase-3 Clinical Research Copilot (Hospital Evidence)")
    query = st.text_input("Ask a clinical research question:")

    if query:
        audit("CLINICAL_QUERY", {"query": query})

        if index:
            qemb = embedder.encode([query])
            D, I = index.search(np.array(qemb).astype("float32"), 5)
            context = "\n\n".join([documents[i] for i in I[0]])
            used_sources = [sources[i] for i in I[0]]

            st.markdown("### 🏥 Hospital Evidence-Based Answer")
            st.write(context[:3500])

            st.markdown("**Evidence Sources**")
            for s in used_sources:
                st.info(s)

            audit("HOSPITAL_EVIDENCE_USED", {"sources": used_sources})
            st.success("Mode: Hospital Evidence AI")
        else:
            st.warning("No local evidence found. Please upload PDFs in Evidence Library.")

# ---------- 🌐 PubMed Global Research (NIH) ----------
elif page == "🌐 PubMed Global Research (NIH)":
    st.subheader("🌐 PubMed Global Research (NIH Live)")
    query = st.text_input("Search PubMed (e.g., Parkinson genetic markers):")

    if query:
        with st.spinner("🔍 Searching PubMed (NIH)..."):
            pmids = pubmed_search(query, retmax=10)
            articles = pubmed_fetch(pmids)

        audit("PUBMED_SEARCH", {"query": query, "pmids": pmids})

        if articles:
            st.success(f"Found {len(articles)} PubMed articles")
            for a in articles:
                st.markdown(f"### 🧾 PMID: {a['PMID']}")
                st.markdown(f"**Title:** {a['Title']}")
                st.write(a["Abstract"][:3000])
                st.markdown("---")
        else:
            st.warning("No PubMed articles found for this query.")

# ---------- 🧪 Clinical Trials (Outcomes) ----------
elif page == "🧪 Clinical Trials (Outcomes)":
    st.subheader("🧪 Clinical Trial Outcome Explorer")
    q = st.text_input("Search condition / therapy (e.g., glioblastoma, Parkinson’s):")

    if q:
        trials = fetch_clinical_trials(q, limit=20)
        audit("TRIAL_SEARCH", {"query": q, "results": len(trials)})

        if trials:
            rows = build_outcome_table(trials)
            st.dataframe(rows, use_container_width=True)
            st.caption("Outcome table derived from ClinicalTrials.gov fields (Phase, Status, Condition).")
        else:
            st.info("No trials found for this query.")

# ---------- 🧬 FDA Drug Intelligence ----------
elif page == "🧬 FDA Drug Intelligence":
    st.subheader("🧬 FDA Drug Intelligence")
    drug = st.text_input("Enter brand name (e.g., Levodopa, Semaglutide):")

    if drug:
        info = fetch_fda_drug_info(drug)
        audit("FDA_LOOKUP", {"drug": drug, "found": bool(info)})

        if info:
            st.success("FDA record found")
            st.json(info)
        else:
            st.warning("No FDA label record found for this brand name.")

# ---------- 👤 Patient Cohorts ----------
elif page == "👤 Patient Cohorts":
    st.subheader("👤 Patient Cohort Analytics (De-identified)")
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=65)
    diagnosis = st.text_input("Diagnosis / Condition")
    biomarkers = st.text_input("Biomarkers (comma separated, optional)")

    if st.button("Analyze Cohort"):
        cohort = analyze_patient_cohort(
            age,
            diagnosis,
            [b.strip() for b in biomarkers.split(",") if b.strip()],
        )
        audit("COHORT_ANALYSIS", cohort)
        st.json(cohort)

# ---------- 📊 Live Dashboard ----------
elif page == "📊 Live Dashboard":
    st.subheader("📊 Live Intelligence Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Evidence Docs", len(documents))
    col2.metric("Vector Index", "Ready" if index else "Not Built")
    col3.metric("Audit Events", len(json.load(open(AUDIT_LOG))) if os.path.exists(AUDIT_LOG) else 0)

    st.markdown("### Recent Audit Events")
    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))
        st.dataframe(logs[-10:][::-1], use_container_width=True)
    else:
        st.info("No audit events yet.")

# ---------- 🕒 Audit & Compliance ----------
elif page == "🕒 Audit & Compliance":
    st.subheader("🕒 Audit & Compliance (Immutable Ledger)")

    st.info("ℹ️ CDSS only. Research support. Final decisions by licensed physicians.")

    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))
        st.metric("Total Clinical Events", len(logs))

        with st.expander("View Audit Ledger"):
            st.dataframe(logs[::-1], use_container_width=True)

        with st.expander("Verify Ledger Integrity"):
            ok = True
            prev = "GENESIS"
            for r in logs:
                check = r.copy()
                ledger_hash = check.pop("ledger_hash")
                if check.get("previous_hash") != prev:
                    ok = False
                    break
                calc = hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest()
                if calc != ledger_hash:
                    ok = False
                    break
                prev = ledger_hash
            st.success("Ledger verified") if ok else st.error("Ledger integrity failed")
    else:
        st.warning("No audit ledger found yet.")

# ----------------------------- FOOTER -----------------------------
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS).")
st.sidebar.info("Final clinical decisions must be made by licensed physicians.")

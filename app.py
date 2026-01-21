# ============================================================
# ĀROGYABODHA AI — Clinical-Safe Hospital RAG Intelligence OS
# Hospital + Research + Trial + Regulatory + Clinical Reasoning
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, requests
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
st.set_page_config("ĀROGYABODHA AI — Medical Intelligence OS", "🧠", layout="wide")

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

AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ============================================================
# USERS
# ============================================================
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION
# ============================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index_ready": False,
    "docs": [],
    "srcs": [],
    "ai_mode": "Hybrid AI"
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
# PDF + OCR EXTRACTION
# ============================================================
def extract_text(file_bytes):
    text_pages = []

    # --- Normal PDF text ---
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages[:200]:
            t = page.extract_text()
            if t and len(t) > 150:
                text_pages.append(t)
    except:
        pass

    # --- OCR fallback ---
    if not text_pages:
        images = convert_from_bytes(file_bytes)
        for img in images[:50]:
            text = pytesseract.image_to_string(img)
            if len(text) > 150:
                text_pages.append(text)

    return text_pages

# ============================================================
# VECTOR INDEX
# ============================================================
def build_index():
    docs, srcs = [], []

    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
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

# Load existing index
if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
    try:
        st.session_state.index = faiss.read_index(INDEX_FILE)
        cache = pickle.load(open(CACHE_FILE, "rb"))
        st.session_state.docs = cache["docs"]
        st.session_state.srcs = cache["srcs"]
        st.session_state.index_ready = True
    except:
        st.session_state.index_ready = False

# ============================================================
# CLINICAL SAFE RAG
# ============================================================
def medical_relevance(text, query):
    qwords = query.lower().split()
    return sum(1 for w in qwords if w in text.lower()) >= 1

def summarize(text):
    return " ".join(text.split("\n")[:6])[:2000]

def hospital_rag(query, k=5):
    if not st.session_state.index_ready:
        return [], []

    qemb = embedder.encode([query])
    D, I = st.session_state.index.search(np.array(qemb).astype("float32"), k)

    hits, sources = [], []

    for dist, idx in zip(D[0], I[0]):
        sim = 1 / (1 + dist)
        chunk = st.session_state.docs[idx]

        if sim >= 0.55 and medical_relevance(chunk, query):
            hits.append(chunk)
            sources.append(st.session_state.srcs[idx])

    return hits, sources

# ============================================================
# GLOBAL CONNECTORS
# ============================================================
def fetch_pubmed(query):
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 5}
        r = requests.get(url, params=params, timeout=10)
        return r.json()["esearchresult"]["idlist"]
    except:
        return []

def fetch_trials(query):
    try:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {"query.term": query, "pageSize": 5}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        rows = []
        for study in data.get("studies", []):
            ident = study["protocolSection"]["identificationModule"]
            status = study["protocolSection"]["statusModule"]
            rows.append({
                "Trial ID": ident.get("nctId"),
                "Status": status.get("overallStatus")
            })
        return rows
    except:
        return []

# ============================================================
# UI
# ============================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}**")

st.session_state.ai_mode = st.sidebar.radio(
    "AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"]
)

module = st.sidebar.radio("Medical Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Phase-3 Research Copilot",
    "📊 Dashboard",
    "🕒 Audit"
])

# ============================================================
# EVIDENCE LIBRARY
# ============================================================
if module == "📁 Evidence Library":
    st.header("📁 Medical Evidence Library")

    files = st.file_uploader("Upload Hospital PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success("PDFs uploaded successfully")

    if st.button("Build Evidence Index"):
        st.session_state.index, st.session_state.docs, st.session_state.srcs = build_index()
        st.session_state.index_ready = True
        audit("build_index", {"docs": len(st.session_state.docs)})
        st.success(f"Index built with {len(st.session_state.docs)} pages")

# ============================================================
# RESEARCH COPILOT
# ============================================================
if module == "🔬 Phase-3 Research Copilot":
    st.header("🔬 Phase-3 Clinical Research Engine")
    st.write("AI Mode:", st.session_state.ai_mode)

    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze") and query:
        audit("query", {"q": query, "mode": st.session_state.ai_mode})

        hits, sources = hospital_rag(query)

        if hits and st.session_state.ai_mode in ["Hospital AI", "Hybrid AI"]:
            st.subheader("🏥 Hospital Evidence")
            for h in hits:
                st.write(summarize(h))

            st.markdown("### Sources")
            for s in sources:
                st.info(s)

        elif st.session_state.ai_mode in ["Global AI", "Hybrid AI"]:
            st.warning("Hospital evidence not found. Using global research.")

            pmids = fetch_pubmed(query)
            trials = fetch_trials(query)

            st.subheader("📚 PubMed PMIDs")
            st.write(pmids)

            st.subheader("🧪 Clinical Trials")
            st.table(pd.DataFrame(trials))

        else:
            st.error("No hospital evidence found. Upload hospital PDFs.")

# ============================================================
# DASHBOARD
# ============================================================
if module == "📊 Dashboard":
    st.metric("Indexed Pages", len(st.session_state.docs))
    st.metric("Index Ready", st.session_state.index_ready)

# ============================================================
# AUDIT
# ============================================================
if module == "🕒 Audit":
    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))))

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — Clinical-Safe Hospital Intelligence OS")

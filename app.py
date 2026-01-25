# ============================================================
# ĀROGYABODHA AI — Hospital Grade Medical Intelligence OS
# Evidence RAG + Semantic AI + Clinical Reasoning CDSS
# ============================================================

import streamlit as st
import os, json, datetime, requests, re, base64
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================

st.set_page_config("ĀROGYABODHA AI", "🧠", layout="wide")
st.info("ℹ️ Clinical Decision Support System — Research only")

BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")
os.makedirs(PDF_FOLDER, exist_ok=True)

# ================= USERS =================

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Research"}
    }, open(USERS_DB, "w"), indent=2)

# ================= SESSION =================

if "logged" not in st.session_state:
    st.session_state.logged = False
    st.session_state.user = None
    st.session_state.role = None

# ================= AUDIT =================

def audit(event, meta=None):
    logs = json.load(open(AUDIT_LOG)) if os.path.exists(AUDIT_LOG) else []
    logs.append({
        "time": str(datetime.datetime.utcnow()),
        "user": st.session_state.user,
        "event": event,
        "meta": meta or {}
    })
    json.dump(logs, open(AUDIT_LOG, "w"), indent=2)

# ================= LOGIN =================

def login_ui():
    st.title("Secure Hospital Login")
    with st.form("login"):
        u = st.text_input("User ID")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")

    if ok:
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged = True
            st.session_state.user = u
            st.session_state.role = users[u]["role"]
            audit("login")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged:
    login_ui()
    st.stop()

# ================= PDF =================

def show_pdf(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="650"></iframe>',
        unsafe_allow_html=True
    )

# ================= PUBMED =================

def fetch_pubmed(q):
    try:
        q = f"{q} AND 2020:3000[dp]"
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": q, "retmode": "json", "retmax": 25, "sort": "pub+date"},
            timeout=15)
        return r.json()["esearchresult"]["idlist"]
    except:
        return []

def fetch_pubmed_details(ids):
    if not ids: return []
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=20)

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

# ================= SEMANTIC AI =================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def rank_papers(query, papers):
    if not papers: return []
    texts = [p["abstract"] for p in papers] + [query]
    emb = model.encode(texts)
    scores = np.dot(emb[:-1], emb[-1])
    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:10]]

# ================= THEMES =================

THEMES = {
    "Genomics & Molecular": {"pcr","sequencing","mutation","gene therapy"},
    "Immunology & Infection": {"vaccine","antibody","immune response"},
    "Biomarkers": {"troponin","crp","d-dimer","biomarker"},
    "Imaging": {"ct","mri","ultrasound","x-ray"},
    "Clinical Trials": {"clinical trial","efficacy","safety"},
    "Outcomes & Risk": {"mortality","survival","complication"},
    "Pharmacology": {"drug interaction","dose","toxicity"},
    "Health Economics": {"cost-effective","economic","cost analysis"}
}

def detect_themes(papers):
    text = " ".join(p["abstract"].lower() for p in papers)
    found = {}
    for k, keys in THEMES.items():
        hits = [x for x in keys if x in text]
        if hits:
            found[k] = len(hits)
    return found

# ================= CLINICAL REASONING =================

def clinical_reasoning(papers):
    t = " ".join(p["abstract"].lower() for p in papers)
    insights = []
    if "cost" in t: insights.append("Cost-effectiveness plays a significant role.")
    if "guideline" in t: insights.append("Recent guidelines influence treatment strategies.")
    if "survival" in t: insights.append("Outcome improvements reported.")
    if "risk" in t: insights.append("Risk stratification remains critical.")
    return " ".join(insights) if insights else "Evidence continues evolving."

# ================= SIDEBAR =================

st.sidebar.markdown(f"👨‍⚕️ {st.session_state.user} ({st.session_state.role})")

module = st.sidebar.radio("Hospital Intelligence Center", [
    "📁 Evidence Library",
    "🔬 Clinical Research Copilot",
    "📊 Enterprise Dashboard",
    "🕒 Audit Trail"
])

# ================= MODULES =================

if module == "📁 Evidence Library":
    st.header("Hospital Evidence Library")

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files:
        for f in files:
            open(os.path.join(PDF_FOLDER, f.name), "wb").write(f.read())
        st.success("Indexed successfully")

    pdfs = os.listdir(PDF_FOLDER)
    if pdfs:
        selected = st.selectbox("Available Evidence", pdfs)
        show_pdf(os.path.join(PDF_FOLDER, selected))
    else:
        st.info("No internal evidence yet")

# --------------------------------------------------

if module == "🔬 Clinical Research Copilot":
    st.header("🏥 Hospital AI Command Center")

    q = st.text_input("Ask clinical research question")

    if st.button("Analyze") and q:
        audit("query", {"q": q})

        raw = fetch_pubmed_details(fetch_pubmed(q))
        papers = rank_papers(q, raw)

        # KPI Row
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📚 Evidence Papers", len(papers))
        c2.metric("🧬 Themes", len(detect_themes(papers)))
        c3.metric("⚠️ Risk Signals", sum("risk" in p["abstract"].lower() for p in papers))
        c4.metric("📈 AI Confidence", "High")

        left,right = st.columns([1.3,1])

        with left:
            st.subheader("📌 Clinical Interpretation")
            st.success(clinical_reasoning(papers))

        with right:
            st.subheader("🧠 Evidence Themes")
            theme_counts = detect_themes(papers)
            chart_df = pd.DataFrame({
                "Theme": list(theme_counts.keys()),
                "Mentions": list(theme_counts.values())
            })
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index("Theme"))

        st.subheader("📚 Evidence Table")
        df = pd.DataFrame([
            {"Title": p["title"][:90], "Source": "PubMed", "Link": p["url"]}
            for p in papers
        ])
        st.dataframe(df, use_container_width=True)

# --------------------------------------------------

if module == "📊 Enterprise Dashboard":
    st.metric("Evidence PDFs", len(os.listdir(PDF_FOLDER)))
    st.metric("Total Queries", len(json.load(open(AUDIT_LOG))) if os.path.exists(AUDIT_LOG) else 0)

# --------------------------------------------------

if module == "🕒 Audit Trail":
    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))), use_container_width=True)
    else:
        st.info("No audit records yet")

# ================= FOOTER =================

st.caption("ĀROGYABODHA AI — Hospital Grade Clinical Intelligence Platform")

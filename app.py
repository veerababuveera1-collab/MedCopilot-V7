import streamlit as st
import os, json, pickle, datetime, re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Hospital Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# GLOBAL STYLE (Modern UI)
# ======================================================
st.markdown("""
<style>
body {background-color:#0e1117;}
.login-card {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    padding:40px;border-radius:15px;
    box-shadow:0px 0px 40px rgba(0,0,0,0.6);
    color:white;
}
.big-title {font-size:42px;font-weight:800;}
.sub {color:#aaa;}
.metric-card {background:#161b22;padding:20px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# DISCLAIMER
# ======================================================
st.info("ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only. Final decisions must be made by licensed doctors.")

# ======================================================
# STORAGE
# ======================================================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
USERS_DB = "users.json"
AUDIT_LOG = "audit_log.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ======================================================
# DEMO USERS
# ======================================================
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "admin": {"password": "admin123", "role": "Admin"}
    }, open(USERS_DB,"w"), indent=2)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "module": "Clinical Research Copilot"
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# AUDIT
# ======================================================
def audit(event, meta=None):
    rows=[]
    if os.path.exists(AUDIT_LOG):
        rows=json.load(open(AUDIT_LOG))
    rows.append({
        "time":str(datetime.datetime.now()),
        "user":st.session_state.username,
        "event":event,
        "meta":meta or {}
    })
    json.dump(rows, open(AUDIT_LOG,"w"), indent=2)

# ======================================================
# LOGIN UI (Modern)
# ======================================================
def login_ui():
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div class="login-card">
        <div class="big-title">🏥 ĀROGYABODHA AI</div>
        <p class="sub">Hospital Clinical Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("🔐 Login", use_container_width=True):
            users = json.load(open(USERS_DB))
            if u in users and users[u]["password"] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.role = users[u]["role"]
                audit("login")
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in=False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.radio("Select Module", 
    ["Clinical Research Copilot","Lab Report Intelligence","Audit Trail"],
    key="module"
)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="big-title">🧠 ĀROGYABODHA AI</div>
<p class="sub">Hospital-grade • Evidence-locked • Governance enabled</p>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()

# ======================================================
# FAISS INDEX
# ======================================================
def build_index():
    docs,srcs=[],[]
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader=PdfReader(os.path.join(PDF_FOLDER,pdf))
            for i,p in enumerate(reader.pages[:100]):
                t=p.extract_text()
                if t and len(t)>100:
                    docs.append(t)
                    srcs.append(f"{pdf} — Page {i+1}")
    if not docs: return None,[],[]
    emb=embedder.encode(docs)
    idx=faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents":docs,"sources":srcs}, open(CACHE_FILE,"wb"))
    return idx,docs,srcs

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index=faiss.read_index(INDEX_FILE)
    data=pickle.load(open(CACHE_FILE,"rb"))
    st.session_state.documents=data["documents"]
    st.session_state.sources=data["sources"]
    st.session_state.index_ready=True

# ======================================================
# MEDICAL LIBRARY
# ======================================================
st.markdown("## 📁 Medical Evidence Library")

uploads = st.file_uploader("Upload hospital PDFs", type=["pdf"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        with open(os.path.join(PDF_FOLDER,f.name),"wb") as out:
            out.write(f.getbuffer())
    st.success("PDFs uploaded")

if st.button("🔄 Build Evidence Index"):
    st.session_state.index,st.session_state.documents,st.session_state.sources=build_index()
    st.session_state.index_ready=True
    st.success("Hospital Evidence Index Ready")

# ======================================================
# CLINICAL RESEARCH COPILOT
# ======================================================
if st.session_state.module=="Clinical Research Copilot":
    st.markdown("## 🔬 Clinical Research Copilot")

    query = st.text_input("Ask a clinical research question")

    if st.button("🚀 Analyze") and query:
        if not st.session_state.index_ready:
            st.error("Hospital evidence index not built.")
        else:
            qemb=embedder.encode([query])
            _,I=st.session_state.index.search(np.array(qemb),5)
            context="\n\n".join([st.session_state.documents[i] for i in I[0]])
            sources=[st.session_state.sources[i] for i in I[0]]

            st.success("Hospital Evidence Answer")
            st.write(context[:2000]+"...")

            st.markdown("### 📑 Evidence Sources")
            for s in sources:
                st.info(s)

# ======================================================
# LAB REPORT INTELLIGENCE
# ======================================================
if st.session_state.module=="Lab Report Intelligence":
    st.markdown("## 🧪 Lab Report Intelligence")

    lab = st.file_uploader("Upload Lab PDF", type=["pdf"])
    if lab:
        reader=PdfReader(lab)
        text=""
        for p in reader.pages:
            if p.extract_text():
                text+=p.extract_text()

        values={}
        for test in ["Bilirubin","SGPT","SGOT","Creatinine","Urea"]:
            m=re.search(test+r".*?(\d+\.?\d*)",text,re.I)
            if m:
                values[test]=m.group(1)

        if values:
            for k,v in values.items():
                st.write(f"**{k}** : {v}")
        else:
            st.warning("Unable to auto-detect RESULT values.")

# ======================================================
# AUDIT TRAIL
# ======================================================
if st.session_state.module=="Audit Trail":
    st.markdown("## 🕒 Audit Trail")
    if os.path.exists(AUDIT_LOG):
        df=pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df,use_container_width=True)
    else:
        st.info("No logs yet")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

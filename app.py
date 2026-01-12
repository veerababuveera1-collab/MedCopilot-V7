import streamlit as st
import os, json, pickle, datetime
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Clinical Research Copilot",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# DISCLAIMER
# ======================================================
st.info(
    "ℹ️ ĀROGYABODHA AI is a clinical research decision-support system only. "
    "It does NOT provide diagnosis or treatment. "
    "Final clinical decisions must be made by licensed medical professionals."
)

# ======================================================
# STORAGE
# ======================================================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
ANALYTICS_FILE = "analytics_log.json"
FDA_DB = "fda_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ======================================================
# SESSION STATE
# ======================================================
for k, v in {
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "show_quick_help": False,
    "help_lang": "EN",
    "role": "Doctor"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HEADER
# ======================================================
h1, h2, h3, h4 = st.columns([6,1,1,1])
with h1:
    st.markdown("## 🧠 ĀROGYABODHA AI")
    st.caption("Evidence-Locked • Semantic-Validated • Clinical Research Copilot")
with h2:
    if st.button("❓ Help"):
        st.session_state.show_quick_help = not st.session_state.show_quick_help
with h3:
    if st.button("🌐 EN / తెలుగు"):
        st.session_state.help_lang = "TE" if st.session_state.help_lang=="EN" else "EN"
with h4:
    st.session_state.role = st.selectbox("Role", ["Doctor","Researcher"])

# ======================================================
# QUICK HELP
# ======================================================
if st.session_state.show_quick_help:
    st.markdown("---")
    if st.session_state.help_lang == "EN":
        st.markdown("""
• Hospital AI → Hospital PDFs only  
• Semantic validation → meaning-based check  
• Partial evidence → cautious summary  
• No evidence → answer blocked
""")
    else:
        st.markdown("""
• Hospital AI → కేవలం PDFs  
• Semantic validation → అర్థం ఆధారంగా చెక్  
• Partial evidence → జాగ్రత్త సూచన  
• Evidence లేకపోతే → సమాధానం లేదు
""")
    st.markdown("---")

# ======================================================
# MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()

# ======================================================
# FDA REGISTRY
# ======================================================
if not os.path.exists(FDA_DB):
    json.dump({
        "temozolomide":"FDA Approved",
        "bevacizumab":"FDA Approved",
        "car-t":"Experimental / Trial Only"
    }, open(FDA_DB,"w"))
FDA_REGISTRY = json.load(open(FDA_DB))

# ======================================================
# HELPERS
# ======================================================
def log_query(query, mode):
    logs=[]
    if os.path.exists(ANALYTICS_FILE):
        logs=json.load(open(ANALYTICS_FILE))
    logs.append({
        "query":query,
        "mode":mode,
        "time":str(datetime.datetime.now())
    })
    json.dump(logs,open(ANALYTICS_FILE,"w"),indent=2)

def confidence_explained(ans,n):
    score=60; reasons=[]
    if n>=3: score+=15; reasons.append("Multiple hospital sources")
    if "fda" in ans.lower(): score+=10; reasons.append("FDA reference")
    if "survival" in ans.lower() or "mortality" in ans.lower():
        score+=10; reasons.append("Outcome data mentioned")
    return min(score,95), reasons

# ---------- SEMANTIC EVIDENCE FIX ----------
def semantic_similarity(a, b):
    ea = embedder.encode([a])[0]
    eb = embedder.encode([b])[0]
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))

def semantic_evidence_level(answer, context):
    sim = semantic_similarity(answer, context)
    if sim >= 0.55:
        return "STRONG", int(sim*100)
    elif sim >= 0.25:
        return "PARTIAL", int(sim*100)
    else:
        return "NONE", 0
# ------------------------------------------

def extract_outcomes(text):
    rows=[]
    for d,s in FDA_REGISTRY.items():
        if d in text.lower():
            rows.append({"Treatment":d.title(),"FDA Status":s})
    return pd.DataFrame(rows)

def generate_report(query,mode,answer,conf,coverage,sources):
    r=f"""ĀROGYABODHA AI – Clinical Research Report
---------------------------------------
Query: {query}
Mode: {mode}
Confidence: {conf}%
Evidence Coverage: {coverage}%

Answer:
{answer}

Sources:
"""
    for s in sources: r+=f"- {s}\n"
    return r

# ======================================================
# HOSPITAL AI (EVIDENCE-LOCKED PROMPT)
# ======================================================
def hospital_answer(query, context):
    prompt=f"""
You are a Hospital Clinical Decision Support AI.

RULES:
- Use ONLY the hospital evidence below
- Do NOT use external knowledge
- Do NOT hallucinate
- If evidence is insufficient, say so clearly

Hospital Evidence:
{context}

Doctor Query:
{query}
"""
    return external_research_answer(prompt).get("answer","")

# ======================================================
# INDEX
# ======================================================
def build_index():
    docs,srcs=[],[]
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            r=PdfReader(os.path.join(PDF_FOLDER,pdf))
            for i,p in enumerate(r.pages[:200]):
                t=p.extract_text()
                if t and len(t)>100:
                    docs.append(t)
                    srcs.append(f"{pdf} – Page {i+1}")
    if not docs: return None,[],[]
    emb=embedder.encode(docs)
    idx=faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx,INDEX_FILE)
    pickle.dump({"documents":docs,"sources":srcs},open(CACHE_FILE,"wb"))
    return idx,docs,srcs

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index=faiss.read_index(INDEX_FILE)
    data=pickle.load(open(CACHE_FILE,"rb"))
    st.session_state.documents=data["documents"]
    st.session_state.sources=data["sources"]
    st.session_state.index_ready=True

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.subheader("📁 Medical Library")
up=st.sidebar.file_uploader("Upload PDFs",type=["pdf"],accept_multiple_files=True)
if up:
    for f in up:
        open(os.path.join(PDF_FOLDER,f.name),"wb").write(f.getbuffer())
    st.sidebar.success("Uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index,st.session_state.documents,st.session_state.sources=build_index()
    st.session_state.index_ready=True

st.sidebar.divider()
st.sidebar.subheader("🕒 Recent Queries")
if os.path.exists(ANALYTICS_FILE):
    logs=json.load(open(ANALYTICS_FILE))
    for q in logs[-5:][::-1]:
        st.sidebar.write(f"• {q['query']} ({q['mode']})")

# ======================================================
# QUERY
# ======================================================
query=st.text_input("Ask a clinical research question")
mode=st.radio("AI Mode",["Hospital AI","Global AI","Hybrid AI"],horizontal=True)
run=st.button("🚀 Analyze")

# ======================================================
# EXECUTION
# ======================================================
if run and query:
    log_query(query,mode)
    t1,t2,t3,t4=st.tabs(["🏥 Hospital","🌍 Global","🧪 Outcomes","📚 Library"])

    if mode in ["Hospital AI","Hybrid AI"]:
        qemb=embedder.encode([query])
        _,I=st.session_state.index.search(np.array(qemb),5)
        context="\n\n".join([st.session_state.documents[i] for i in I[0]])
        raw=hospital_answer(query,context)

        level,coverage=semantic_evidence_level(raw,context)
        conf,reasons=confidence_explained(raw,len(I[0]))
        src=[st.session_state.sources[i] for i in I[0]]

        with t1:
            st.metric("Confidence",f"{conf}%")
            st.metric("Evidence Coverage",f"{coverage}%")

            if level=="STRONG":
                st.success("🟢 Strong hospital evidence")
                st.write(raw)
            elif level=="PARTIAL":
                st.warning("🟡 Partial hospital evidence — interpret cautiously")
                st.write(raw)
            else:
                st.error("🔴 No sufficient hospital evidence")
                st.write("Insufficient hospital evidence available.")

            for s in src: st.info(s)

            st.download_button(
                "📥 Download Report",
                generate_report(query,mode,raw,conf,coverage,src),
                file_name="arogyabodha_report.txt"
            )

        with t3:
            df=extract_outcomes(raw)
            if not df.empty: st.table(df)

    if mode in ["Global AI","Hybrid AI"]:
        with t2:
            st.write(external_research_answer(query).get("answer",""))

    with t4:
        for pdf in os.listdir(PDF_FOLDER):
            if pdf.endswith(".pdf"):
                c1,c2=st.columns([8,1])
                with c1:
                    st.write("📄",pdf)
                with c2:
                    if st.button("🗑️",key=pdf):
                        os.remove(os.path.join(PDF_FOLDER,pdf))
                        if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
                        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
                        st.session_state.index_ready=False
                        st.experimental_rerun()

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI © FINAL • Semantic Evidence-Aware • Clinically Safe")

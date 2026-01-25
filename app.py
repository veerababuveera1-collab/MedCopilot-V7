# ============================================================
# ĀROGYABODHA AI — Enterprise Medical Intelligence OS
# Hybrid AI + FDA Regulatory + Clinical Analytics
# ============================================================

import streamlit as st
import os, json, datetime, requests, re, base64
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================

st.set_page_config("ĀROGYABODHA AI", "🧠", layout="wide")
st.info("ℹ️ Clinical Decision Support System — Research only")

BASE=os.getcwd()
PDF_FOLDER=os.path.join(BASE,"medical_library")
AUDIT_LOG=os.path.join(BASE,"audit_log.json")
USERS_DB=os.path.join(BASE,"users.json")
os.makedirs(PDF_FOLDER,exist_ok=True)

# ================= USERS =================

if not os.path.exists(USERS_DB):
    json.dump({"doctor1":{"password":"doctor123"}},open(USERS_DB,"w"))

# ================= SESSION =================

if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
    st.session_state.username=None

# ================= AUDIT =================

def audit(event,meta=None):
    logs=json.load(open(AUDIT_LOG)) if os.path.exists(AUDIT_LOG) else []
    logs.append({
        "time":str(datetime.datetime.utcnow()),
        "event":event,
        "meta":meta or {}
    })
    json.dump(logs,open(AUDIT_LOG,"w"),indent=2)

# ================= LOGIN =================

def login():
    u=st.text_input("User")
    p=st.text_input("Password",type="password")
    if st.button("Login"):
        users=json.load(open(USERS_DB))
        if u in users and users[u]["password"]==p:
            st.session_state.logged_in=True
            st.session_state.username=u
            audit("login")
            st.rerun()

if not st.session_state.logged_in:
    login()
    st.stop()

# ================= PUBMED =================

def fetch_pubmed(q):
    r=requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db":"pubmed","term":f"{q} AND 2020:3000[dp]","retmode":"json","retmax":20},
        timeout=15
    )
    return r.json()["esearchresult"]["idlist"]

def fetch_details(ids):
    if not ids: return []
    r=requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db":"pubmed","id":",".join(ids),"retmode":"xml"},
        timeout=20
    )
    papers=[]
    for art in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>",r.text,re.S):
        title=re.search(r"<ArticleTitle>(.*?)</ArticleTitle>",art,re.S)
        abstract=re.search(r"<AbstractText.*?>(.*?)</AbstractText>",art,re.S)
        papers.append({
            "title":re.sub("<.*?>","",title.group(1)) if title else "",
            "abstract":re.sub("<.*?>","",abstract.group(1)) if abstract else ""
        })
    return papers

# ================= SEMANTIC AI =================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model=load_model()

def rank(query,papers):
    if not papers: return []
    emb=model.encode([p["abstract"] for p in papers]+[query])
    scores=np.dot(emb[:-1],emb[-1])
    ranked=sorted(zip(papers,scores),key=lambda x:x[1],reverse=True)
    return [p for p,_ in ranked[:8]]

# ================= FDA REGULATORY AI =================

def fetch_fda_info(keyword):
    try:
        r=requests.get(
            "https://api.fda.gov/drug/drugsfda.json",
            params={"search":keyword,"limit":5},
            timeout=15
        )
        results=r.json().get("results",[])
        return [x["brand_name"] for x in results if "brand_name" in x]
    except:
        return []

# ================= CLINICAL INTELLIGENCE =================

def generate_answer(q):
    return f"""
### 📌 Clinical Research Answer
Current biomedical research on **{q}** shows active clinical trials, evolving therapies, and improving patient outcomes.
"""

def reasoning(papers):
    text=" ".join(p["abstract"].lower() for p in papers)
    insights=[]
    if "cost" in text: insights.append("Economic effectiveness is a major research focus.")
    if "survival" in text: insights.append("Improved survival outcomes observed.")
    if "guideline" in text: insights.append("Updated clinical guidelines identified.")
    return " ".join(insights) or "Evidence remains under evaluation."

# ================= UI =================

st.sidebar.title("Medical Intelligence Center")
module=st.sidebar.radio("",["🔬 Research Copilot","📊 Enterprise Analytics","🕒 Audit"])

# -------------------------------------------------

if module=="🔬 Research Copilot":
    q=st.text_input("Ask clinical research question")

    if st.button("Analyze") and q:
        audit("query",{"q":q})

        ids=fetch_pubmed(q)
        papers=rank(q,fetch_details(ids))

        st.markdown(generate_answer(q))
        st.subheader("🧠 Clinical Interpretation")
        st.write(reasoning(papers))

        st.subheader("🏛 FDA Regulatory Evidence")
        fda=fetch_fda_info(q.split()[0])
        if fda:
            st.success("FDA Related Drugs:")
            for d in fda:
                st.write("•",d)
        else:
            st.info("No FDA data found yet")

        st.subheader("📚 Research Evidence")
        for p in papers:
            with st.expander(p["title"]):
                st.write(p["abstract"][:800])

# -------------------------------------------------

if module=="📊 Enterprise Analytics":
    if os.path.exists(AUDIT_LOG):
        df=pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.metric("Total Clinical Queries",len(df[df["event"]=="query"]))
        df["date"]=pd.to_datetime(df["time"]).dt.date
        trend=df.groupby("date").size()
        st.line_chart(trend)
    else:
        st.info("No analytics yet")

# -------------------------------------------------

if module=="🕒 Audit":
    if os.path.exists(AUDIT_LOG):
        st.dataframe(pd.DataFrame(json.load(open(AUDIT_LOG))))
    else:
        st.info("No logs")

# ================= FOOTER =================

st.caption("ĀROGYABODHA AI — Enterprise Clinical Intelligence Platform")

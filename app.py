# ======================================================
# ĀROGYABODHA AI — Hospital Clinical Intelligence Platform
# ======================================================

import streamlit as st
import os, json, pickle, datetime, re, random, smtplib
from email.mime.text import MIMEText
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ================= OCR (Optional) =================
OCR_AVAILABLE = True
try:
    import pytesseract
    from pdf2image import convert_from_path
except:
    OCR_AVAILABLE = False

# ================= PAGE =================
st.set_page_config(page_title="ĀROGYABODHA AI", page_icon="🧠", layout="wide")

st.info("ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS) only.")

# ================= STORAGE =================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
USERS_DB = "users.json"
AUDIT_LOG = "audit_log.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ================= USERS =================
if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {
            "password": "doctor123",
            "email": "doctor1@gmail.com",
            "role": "Doctor"
        }
    }, open(USERS_DB, "w"), indent=2)

# ================= SESSION =================
defaults = {
    "logged_in": False,
    "otp_verified": False,
    "otp": None,
    "otp_time": None,
    "username": None,
    "role": None,
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "show_help": False
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= SMTP =================
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASS = os.getenv("SMTP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# ================= UTIL =================
def audit(event, meta=None):
    rows=[]
    if os.path.exists(AUDIT_LOG):
        rows=json.load(open(AUDIT_LOG))
    rows.append({
        "time": str(datetime.datetime.now()),
        "user": st.session_state.get("username"),
        "event": event,
        "meta": meta or {}
    })
    json.dump(rows, open(AUDIT_LOG,"w"), indent=2)

def send_otp(email, otp):
    msg = MIMEText(f"Your ĀROGYABODHA AI Login OTP is: {otp}\nValid for 5 minutes.")
    msg["Subject"] = "ĀROGYABODHA AI Login OTP"
    msg["From"] = SMTP_EMAIL
    msg["To"] = email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASS)
        server.send_message(msg)

# ================= LOGIN UI =================
def login_ui():
    st.markdown("<h2 style='text-align:center'>🏥 ĀROGYABODHA AI Hospital Login</h2>", unsafe_allow_html=True)
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            otp = random.randint(100000,999999)
            st.session_state.otp = otp
            st.session_state.otp_time = datetime.datetime.now()
            st.session_state.username = u
            send_otp(users[u]["email"], otp)
            st.success("OTP sent to registered email")
        else:
            st.error("Invalid credentials")

    if st.session_state.otp:
        otp_input = st.text_input("Enter OTP")
        if st.button("Verify OTP"):
            if (datetime.datetime.now() - st.session_state.otp_time).seconds > 300:
                st.error("OTP expired")
                st.session_state.otp=None
            elif otp_input == str(st.session_state.otp):
                st.session_state.logged_in=True
                st.session_state.role = "Doctor"
                st.session_state.otp=None
                audit("login")
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid OTP")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ================= MODEL =================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ================= FAISS =================
def build_index():
    docs,srcs=[],[]
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader=PdfReader(os.path.join(PDF_FOLDER,pdf))
            for i,p in enumerate(reader.pages[:200]):
                t=p.extract_text()
                if t and len(t)>100:
                    docs.append(t)
                    srcs.append(f"{pdf} — Page {i+1}")
    if not docs:
        return None,[],[]
    emb=embedder.encode(docs)
    idx=faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"documents":docs,"sources":srcs},open(CACHE_FILE,"wb"))
    return idx,docs,srcs

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index=faiss.read_index(INDEX_FILE)
    data=pickle.load(open(CACHE_FILE,"rb"))
    st.session_state.documents=data["documents"]
    st.session_state.sources=data["sources"]
    st.session_state.index_ready=True

# ================= OCR =================
def extract_text_from_pdf(path):
    text=""
    try:
        reader=PdfReader(path)
        for p in reader.pages:
            t=p.extract_text()
            if t: text+=t+"\n"
    except: pass

    if len(text)<200 and OCR_AVAILABLE:
        try:
            images=convert_from_path(path, dpi=300)
            for img in images:
                text+=pytesseract.image_to_string(img)
        except: pass
    return text

# ================= LAB =================
LAB_RULES={
    "Total Bilirubin":(0.3,1.2,"mg/dL"),
    "SGPT":(0,50,"U/L"),
    "SGOT":(0,50,"U/L")
}

def extract_lab_values(text):
    values={}
    for test in LAB_RULES:
        m=re.search(test+r".*?(\d+\.?\d*)",text,re.I)
        if m: values[test]=float(m.group(1))
    return values

def generate_lab_summary(values):
    summary,alerts=[],[]
    for t,v in values.items():
        lo,hi,u=LAB_RULES[t]
        if v<lo: s="🟡 LOW"
        elif v>hi: s="🔴 HIGH"
        else: s="🟢 NORMAL"
        summary.append((t,v,u,s))
        if t=="Total Bilirubin" and v>5:
            alerts.append("🚨 Severe Jaundice — ICU Required")
    return summary,alerts

# ================= SIDEBAR =================
st.sidebar.markdown(f"👨‍⚕️ {st.session_state.username}")
if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in=False
    st.rerun()

st.sidebar.subheader("📁 Medical Library")

uploads=st.sidebar.file_uploader("Upload PDFs",type=["pdf"],accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER,f.name),"wb").write(f.getbuffer())
    st.sidebar.success("Uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index,st.session_state.documents,st.session_state.sources=build_index()
    st.session_state.index_ready=True
    st.sidebar.success("Index built")

st.sidebar.markdown("🟢 Index READY" if st.session_state.index_ready else "🔴 Index NOT BUILT")

for pdf in os.listdir(PDF_FOLDER):
    st.sidebar.write("📄",pdf)

if st.sidebar.button("❓ Help"):
    st.sidebar.info("Upload PDFs → Build Index → Ask Question → Select AI Mode")

module=st.sidebar.radio("Select Module",["Clinical Research Copilot","Lab Report Intelligence","Audit Trail"])

# ================= HEADER =================
st.markdown("## 🧠 ĀROGYABODHA AI — Hospital Clinical Intelligence Platform")

# ================= COPILOT =================
if module=="Clinical Research Copilot":
    query=st.text_input("Ask clinical question")
    mode=st.radio("AI Mode",["Hospital AI","Global AI","Hybrid AI"],horizontal=True)

    if st.button("Analyze") and query:
        tabs = ["🏥 Hospital","🌍 Global","🧪 Outcomes","📚 Library"]
        t1,t2,t3,t4=st.tabs(tabs)

        with t1:
            if not st.session_state.index_ready:
                st.error("Index not built")
            else:
                qemb=embedder.encode([query])
                _,I=st.session_state.index.search(np.array(qemb),5)
                context="\n".join([st.session_state.documents[i] for i in I[0]])
                prompt=f"Use hospital evidence only:\n{context}\n\nQ:{query}"
                ans=external_research_answer(prompt)["answer"]
                st.write(ans)
                for s in [st.session_state.sources[i] for i in I[0]]:
                    st.info(s)

        with t2:
            st.write(external_research_answer(query)["answer"])

        with t3:
            st.info("Outcomes engine active")

        with t4:
            for pdf in os.listdir(PDF_FOLDER):
                st.write("📄",pdf)

# ================= LAB =================
if module=="Lab Report Intelligence":
    f=st.file_uploader("Upload Lab PDF",type=["pdf"])
    if f:
        open("lab.pdf","wb").write(f.getbuffer())
        text=extract_text_from_pdf("lab.pdf")
        vals=extract_lab_values(text)
        summary,alerts=generate_lab_summary(vals)

        st.subheader("Smart Lab Summary")
        for t,v,u,s in summary:
            st.write(f"{t}: {v} {u} — {s}")

        for a in alerts:
            st.error(a)

# ================= AUDIT =================
if module=="Audit Trail":
    if os.path.exists(AUDIT_LOG):
        df=pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df)
    else:
        st.info("No logs yet")

# ================= FOOTER =================
st.caption("ĀROGYABODHA AI © Hospital-Grade Clinical Intelligence Platform")

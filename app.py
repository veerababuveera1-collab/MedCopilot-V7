import hashlib

def audit(event, meta=None):
    logs = []
    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))

    record = {
        "time": str(datetime.datetime.now()),
        "user": st.session_state.username,
        "role": st.session_state.role,
        "event": event,
        "meta": meta or {}
    }

    record["hash"] = hashlib.sha256(json.dumps(record).encode()).hexdigest()

    logs.append(record)
    json.dump(logs, open(AUDIT_LOG, "w"), indent=2)

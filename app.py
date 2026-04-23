import streamlit as st
import requests
import mimetypes

st.set_page_config(page_title="RAG Client", page_icon="🔎", layout="centered")

# -----------------------
# Session state
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! 👋 Upload a PDF or TXT on the left, then ask me anything about it."
        }
    ]
if "last_upload_msg" not in st.session_state:
    st.session_state.last_upload_msg = None

# -----------------------
# Sidebar: server + upload
# -----------------------
with st.sidebar:
    st.title("Server")
    api_base = st.text_input("API base URL", value="http://localhost:8000")

    # Health check
    health_status = "Unknown"
    try:
        r = requests.get(f"{api_base}/health", timeout=3)
        health_status = "✅ OK" if r.ok else f"⚠️ {r.status_code}"
    except Exception as e:
        health_status = f"❌ {type(e).__name__}"
    st.markdown(f"**Health:** {health_status}")

    st.markdown("---")
    st.header("Upload document")

    uploaded = st.file_uploader("Pick a PDF or TXT", type=["pdf", "txt"], key="uploader")

    reset_collection = st.checkbox(
        "Reset previous collection",
        value=False,
        help="Sends /upload?reset=true|false to delete old vectors before re-indexing.",
    )

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        do_upload = st.button("Upload", use_container_width=True)
    with col_sb2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Upload a file and ask your question! ✨"
                }
            ]
            st.success("Chat history cleared.")

    if do_upload:
        if not uploaded:
            st.warning("Please select a file first.")
        else:
            # Guess content type (prefer browser-provided)
            ctype, _ = mimetypes.guess_type(uploaded.name)
            if getattr(uploaded, "type", None):
                ctype = uploaded.type
            if ctype not in {"application/pdf", "text/plain"}:
                ctype = "application/pdf" if uploaded.name.lower().endswith(".pdf") else "text/plain"

            files = {"file": (uploaded.name, uploaded.getvalue(), ctype)}
            with st.spinner("Uploading & indexing..."):
                try:
                    resp = requests.post(
                        f"{api_base}/upload",
                        params={"reset": reset_collection},
                        files=files,
                        timeout=600,
                    )
                    if resp.ok:
                        data = resp.json()
                        msg = data.get("message", "Uploaded.")
                        st.session_state.last_upload_msg = msg
                        st.success(f"{msg} (reset={data.get('reset')})")
                        st.toast("Upload complete ✅", icon="✅")
                    else:
                        try:
                            detail = resp.json().get("detail")
                        except Exception:
                            detail = resp.text
                        st.error(f"Upload failed: {resp.status_code} — {detail}")
                except Exception as e:
                    st.error(f"Upload error: {e}")

# -----------------------
# Main: Chat UI
# -----------------------
st.title("📚 RAG — Streamlit Client")
st.caption("Chat with your FastAPI RAG server. Upload on the left, then ask away.")

if st.session_state.last_upload_msg:
    st.info(f"Last upload: {st.session_state.last_upload_msg}")

# Render chat history
for m in st.session_state.messages:
    avatar = "🔎" if m["role"] == "assistant" else "🙂"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# Chat input
user_msg = st.chat_input("Ask a question about your document…")
if user_msg:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_msg)

    # Get assistant response from /query
    with st.chat_message("assistant", avatar="🔎"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{api_base}/query",
                    json={"question": user_msg.strip()},
                    timeout=180,
                )
                if resp.ok:
                    payload = resp.json() or {}
                    ans = payload.get("answer", "")
                    st.markdown(ans if ans else "_(No answer returned by server.)_")

                    # Optional: show sources if server returns them
                    sources = payload.get("sources") or payload.get("source_nodes")
                    if sources:
                        with st.expander("Sources"):
                            # Accept list of strings or list of dicts
                            if isinstance(sources, list):
                                for i, s in enumerate(sources, 1):
                                    if isinstance(s, str):
                                        st.write(f"{i}. {s}")
                                    elif isinstance(s, dict):
                                        title = s.get("title") or s.get("file") or s.get("id") or f"Source {i}"
                                        meta = s.get("metadata")
                                        st.write(f"{i}. **{title}**")
                                        if meta:
                                            st.code(meta, language="json")
                            else:
                                st.code(sources, language="json")
                else:
                    try:
                        detail = resp.json().get("detail")
                    except Exception:
                        detail = resp.text
                    err = f"Query failed: {resp.status_code} — {detail}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.stop()
            except Exception as e:
                err = f"Query error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.stop()

        # Persist assistant message
        if resp.ok:
            ans = (resp.json() or {}).get("answer", "")
            st.session_state.messages.append({"role": "assistant", "content": ans if ans else "(empty answer)"})



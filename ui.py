import streamlit as st
import requests
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()

# --- CONFIGURATION ---
FASTAPI_URL = "http://127.0.0.1:8000"
INGEST_PASSWORD = os.getenv('INGEST_PASSWORD')

# --- PAGE SETUP ---
st.set_page_config(
    page_title="RAG Agent UI",
    page_icon="🤖",
    layout="wide"
)

### Background
# ⬇️ Add these ~6 lines
# bg = 'https://plus.unsplash.com/premium_vector-1744953414546-12f23070c3d9?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
bg = 'https://images.unsplash.com/photo-1699527381287-06ac070da534?q=80&w=1171&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
st.markdown(
    f"""
    <style>
      .stApp {{
        background: url('{bg}') no-repeat center center fixed;
        background-size: cover;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)





# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ingest Data", "Chat with AI"])

# A SINGLE placeholder we control 100% (no widgets inside; markdown only)
TOOLS_PLACEHOLDER = st.sidebar.empty()

# =========================
# Helpers
# =========================
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        # This informational line is outside the tools placeholder by design
        st.sidebar.info(f"New Thread ID: `{st.session_state.thread_id}`")
    if "last_tool_results" not in st.session_state:
        st.session_state.last_tool_results = []
    if "last_tool_meta" not in st.session_state:
        st.session_state.last_tool_meta = None  # {"updated_at": "...", "question": "..."}

def parse_maybe_json(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

def _md_escape(text: str) -> str:
    # Minimal escaping so Markdown renders safely
    return (
        str(text)
        .replace("\\", r"\\")
        .replace("|", r"\|")
        .replace("_", r"\_")
        .replace("*", r"\*")
        .replace("`", r"\`")
    )

def _to_markdown_table(rows, max_rows: int = 20, max_cols: int = 8) -> str:
    """Render list[dict] as a compact Markdown table."""
    if not rows:
        return "_(empty table)_"
    # determine stable column order
    cols = []
    for r in rows:
        if isinstance(r, dict):
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
    cols = cols[:max_cols] if cols else [f"col{i}" for i in range(min(max_cols, len(rows[0]) if rows else 1))]
    header = "| " + " | ".join(_md_escape(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    body_lines = []
    for r in rows[:max_rows]:
        if not isinstance(r, dict):
            r = {"value": r}
        vals = [_md_escape(r.get(c, "")) for c in cols]
        body_lines.append("| " + " | ".join(vals) + " |")
    more = ""
    if len(rows) > max_rows:
        more = f"\n\n_… {len(rows) - max_rows} more rows omitted_"
    return "\n".join([header, sep, *body_lines]) + more

def _to_markdown_bullets(obj: dict) -> str:
    lines = []
    for k, v in obj.items():
        key = _md_escape(k)
        if isinstance(v, (dict, list)):
            snippet = json.dumps(v, ensure_ascii=False, indent=2)
            if len(snippet) > 1500:
                snippet = snippet[:1500] + "\n… truncated"
            lines.append(f"- **{key}**:\n\n```json\n{snippet}\n```")
        else:
            lines.append(f"- **{key}**: {_md_escape(v)}")
    return "\n".join(lines)

def tool_result_to_markdown(idx: int, tool: dict) -> str:
    name = tool.get("name", "unknown_tool")
    raw  = tool.get("content", "")
    content = parse_maybe_json(raw)

    title = f"### {idx+1}. `{_md_escape(name)}`"
    if isinstance(content, list):
        if content and all(isinstance(x, dict) for x in content):
            body = _to_markdown_table(content)
        else:
            items = "\n".join(f"- {_md_escape(x)}" for x in content[:50])
            more = "" if len(content) <= 50 else f"\n\n_… {len(content)-50} more items omitted_"
            body = items + more
    elif isinstance(content, dict):
        body = _to_markdown_bullets(content)
    else:
        s = str(content)
        looks_json = isinstance(raw, str) and raw.strip().startswith(("{", "["))
        if looks_json:
            s_trim = s[:5000]
            body = f"```json\n{s_trim}\n```" + (" \n_… truncated_" if len(s) > 5000 else "")
        else:
            body = _md_escape(s[:5000]) + (" \n_… truncated_" if len(s) > 5000 else "")
    return f"{title}\n\n{body}"

def build_tools_markdown() -> str:
    meta = st.session_state.get("last_tool_meta")
    results = st.session_state.get("last_tool_results", [])

    header = "## 🛠️ Tools (last turn)"
    meta_lines = []
    if meta:
        q = meta.get("question", "")
        t = meta.get("updated_at", "")
        if q:
            meta_lines.append(f"*Prompt:* _{_md_escape(q)}_")
        if t:
            meta_lines.append(f"*Updated:* {t}")
    meta_md = "\n\n".join(meta_lines) if meta_lines else ""
    if not results:
        # body = "_No tools were used for the last response._"
        body = ""
    else:
        parts = [tool_result_to_markdown(i, t) for i, t in enumerate(results)]
        body = "\n\n---\n\n".join(parts)
    return "\n\n".join([header, meta_md, body]).strip()

def render_sidebar_tool_results():
    # Render PURE markdown into the single placeholder (no widgets), so it NEVER stacks
    TOOLS_PLACEHOLDER.markdown(build_tools_markdown())

def call_ask_api(query: str, thread_id: str):
    ask_url = f"{FASTAPI_URL}/ask"
    payload = {"query": query, "thread_id": thread_id}
    return requests.post(ask_url, json=payload)

# ==================================================================================
# 1. INGESTION PAGE (unchanged)
# ==================================================================================
if page == "Ingest Data":
    st.title("📄 PDF Ingestion")
    st.markdown("Upload a PDF file to process and add it to the vector dense and sparse + knowledge database.")

    with st.form("ingest_form", clear_on_submit=True):
        password = st.text_input("Password (I need to secure the ingest file with a password for my own safety and to prevent abusive use —sorry for the inconvenience 🥰", type="password")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        enable_graph = st.checkbox("Enable Knowledge Graph Upsert", value=True)
        namespace = st.text_input("Namespace", value="default_namespace_1")
        submitted = st.form_submit_button("Ingest File")

        if submitted:
            if password != INGEST_PASSWORD:
                st.error("Incorrect password. Please try again.")
            elif not uploaded_file:
                st.warning("Please upload a PDF file.")
            else:
                ingest_url = f"{FASTAPI_URL}/ingest"
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"enable_graph": enable_graph, "namespace": namespace}

                with st.spinner(f"Processing and ingesting '{uploaded_file.name}'..."):
                    try:
                        response = requests.post(ingest_url, files=files, data=data)
                        if response.status_code == 200:
                            st.success("File ingested successfully!")
                            st.json(response.json())
                        else:
                            st.error(f"Error during ingestion (Status {response.status_code}):")
                            try:
                                st.json(response.json())
                            except json.JSONDecodeError:
                                st.text(response.text)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to connect to the API: {e}")

# ==================================================================================
# 2. CHAT PAGE (markdown sidebar; replace/clear each turn for real)
# ==================================================================================
elif page == "Chat with AI":
    st.title("🤖 Chat with your PDF file")
    st.markdown(
    "Note: I have already ingested a PDF file; please refer to it when asking questions: "
    "[Philippines flood-control corruption article](https://medium.com/@kaikuh/philippines-flood-control-corruption-3e3ce980dfa0)."
)
    
    # Init & render sidebar first so it reflects the current state
    init_state()
    render_sidebar_tool_results()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What is your question?"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 🔴 HARD-CLEAR previous tools immediately (state + UI)
        st.session_state.last_tool_results = []
        st.session_state.last_tool_meta = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": prompt,
        }
        TOOLS_PLACEHOLDER.empty()          # wipe UI completely
        render_sidebar_tool_results()      # re-draw (now shows "No tools...")

        # Call API
        with st.spinner("Thinking..."):
            try:
                resp = call_ask_api(prompt, st.session_state.thread_id)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer found.")
                    tool_results = data.get("tool_results", [])

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # ✅ Replace with latest (or keep empty if none)
                    st.session_state.last_tool_results = tool_results or []
                    st.session_state.last_tool_meta = {
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": prompt,
                    }

                    # Re-render sidebar with the NEW results
                    TOOLS_PLACEHOLDER.empty()
                    render_sidebar_tool_results()
                else:
                    err = f"API Error (Status {resp.status_code}): {resp.text}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

                    # keep cleared
                    st.session_state.last_tool_results = []
                    st.session_state.last_tool_meta = {
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": prompt,
                    }
                    TOOLS_PLACEHOLDER.empty()
                    render_sidebar_tool_results()
            except requests.exceptions.RequestException as e:
                err = f"Connection Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

                # keep cleared
                st.session_state.last_tool_results = []
                st.session_state.last_tool_meta = {
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": prompt,
                }
                TOOLS_PLACEHOLDER.empty()
                render_sidebar_tool_results()

        # Final rerun to keep app consistent after state changes
        st.rerun()

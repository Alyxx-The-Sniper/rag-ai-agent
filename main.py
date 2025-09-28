# api/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from config import DATA_DIR, TMP_DIR
from data_process import parse_single_pdf_to_md, load_markdown_to_documents, chunk_documents
from ingest import get_pinecone_index, create_and_upsert_vectors, upsert_knowledge_graph_from_chunks
from uuid import uuid4
from pathlib import Path
from config import llm_gen
from fastapi.middleware.cors import CORSMiddleware
from agent import get_agent_runnable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from typing import List, Any, Dict, Tuple, Optional,  Union
import uuid



app = FastAPI(title="RAG Ingest API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

################################################################################################################
# 1. INGEST
@app.post("/ingest")
async def run_ingest(
    file: UploadFile = File(...),
    enable_graph: bool = Form(True),
    namespace: str = Form("default_namespace_1"),
):
    """
    Save the uploaded PDF into TMP_DIR, parse to Markdown in TMP_DIR, then index.
    """
    upload_id = str(uuid4())
    safe_stem = Path(file.filename).stem
    pdf_path = DATA_DIR / f"{safe_stem}_{upload_id}.pdf"

    try:
        data = await file.read()
        pdf_path.write_bytes(data)

        # Parse directly into TMP_DIR so .md lives there
        parsed = parse_single_pdf_to_md(pdf_path, TMP_DIR)
        docs = load_markdown_to_documents(parsed)
        chunks = chunk_documents(docs)

        index = get_pinecone_index()
        create_and_upsert_vectors(index, chunks)

        kg = {"enabled": False}
        if enable_graph:
            kg = upsert_knowledge_graph_from_chunks(chunks, llm=llm_gen)

        return {
            "doc_count": len(docs),
            "chunk_count": len(chunks),
            "namespace": namespace,
            "parsed_saved_dir": str(TMP_DIR),
            "parsed_files": [str(p) for p in parsed],
            "kg": kg,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


################################################################################################################
# 2. QUERRY 
# Build once at import (re-uses the same MemorySaver for thread checkpoints)
rag_agent = get_agent_runnable()

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    thread_id: str

class AskResponse(BaseModel):
    answer: str
    tool_results: List[Dict[str, Any]]

class ToolResult(BaseModel):
    name: str
    content: str    


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# from typing import Any, Dict, List
# def extract_tool_messages(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
#     out: List[Dict[str, Any]] = []
#     for m in resp.get("messages", []) or []:
#         t = m.get("type") if isinstance(m, dict) else getattr(m, "type", None)
#         if t == "tool":
#             name = m.get("name", "unknown_tool") if isinstance(m, dict) else (getattr(m, "name", None) or getattr(m, "tool", "unknown_tool"))
#             content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "") or ""
#             out.append({"name": name, "content": content})
#     return out

from typing import Any, Dict, List
from langchain_core.messages import HumanMessage

def extract_tool_messages_last_turn(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return ONLY tool messages that occurred after the most recent HumanMessage.
    This isolates the tools for the latest user question.
    """
    msgs = resp.get("messages", []) or []

    # find the index of the last human message
    last_human_idx = -1
    for i, m in enumerate(msgs):
        # works for dict or BaseMessage
        t = m.get("type") if isinstance(m, dict) else getattr(m, "type", None)
        if t == "human" or isinstance(m, HumanMessage):
            last_human_idx = i

    if last_human_idx == -1:
        return []  # no human found; be safe

    # collect tool messages AFTER that human
    out: List[Dict[str, Any]] = []
    for m in msgs[last_human_idx + 1:]:
        t = m.get("type") if isinstance(m, dict) else getattr(m, "type", None)
        if t == "tool":
            name = (
                m.get("name", "unknown_tool")
                if isinstance(m, dict)
                else (getattr(m, "name", None) or getattr(m, "tool", "unknown_tool"))
            )
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "") or ""
            out.append({"name": name, "content": content})
    return out



# -----------------------------------------------------------------------------
# Endpoint
# -----------------------------------------------------------------------------
# @app.post("/ask", response_model=AskResponse)
# def ask(req: AskRequest): # req.query / req.thread
#     """
#     Invoke the agent with a user query and a thread_id (for persistent memory via MemorySaver).
#     Returns the final answer and the tool outputs used this turn.
#     """

#     initial = {"messages": [HumanMessage(content=req.query)]}
#     config = {"configurable": {"thread_id": req.thread_id}}
#     response = rag_agent.invoke(initial, config=config)

#     answer = response['messages'][-1].content
#     tool_info = extract_tool_messages(response)

#     return AskResponse(answer=answer, tool_results=tool_info)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    initial = {"messages": [HumanMessage(content=req.query)]}
    config = {"configurable": {"thread_id": req.thread_id}}
    response = rag_agent.invoke(initial, config=config)

    answer = response["messages"][-1].content
    tool_info = extract_tool_messages_last_turn(response)  # <-- use new extractor

    return AskResponse(answer=answer, tool_results=tool_info)
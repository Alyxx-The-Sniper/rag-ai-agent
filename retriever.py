"""
Retriver Toold for Langgraph

[Tool 1] = retrieve_and_format_results
- Pinecone Dense + Sparse Retriever (hybrid)
- Cohere ReRank + Contextual Compression wrapper

[Tool 2] = fetch_facts_for_question
- Neo4j Knowledge Graph: fetch facts via full‑text index

"""

# ============================================
# 1. Pinecone Dense and Sparse Retriever
# --------------------------------------------
# Create the hybrid retriever
# ============================================
from pathlib import Path
from typing import Optional, List, Dict, Any
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
import os
import config as cfg
from langchain.tools import tool

###################################################################################
# Helper
def _load_bm25_encoder() -> BM25Encoder:
    """Load BM25 values dumped during ingest (cfg.BM25_PATH or bm25_values.json)."""
    enc = BM25Encoder()
    bm25_path = Path("bm25_values.json")
    return enc.load(str(bm25_path))

# Helper
def _get_pinecone_index():
    """Return a Pinecone Index object (does not create it)."""
    pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
    return pc.Index(cfg.PINECONE_INDEX_NAME)

# Helper
@tool
def build_pinecone_retriever(
    query,
    alpha: float = 0.7,  # 0=sparse only, 1=dense only
    top_k: int = 5,
    # text_key = 'context'
    ) -> List[Dict[str, Any]]:

    """Hybrid Pinecone retriever wrapped with Cohere ReRanker compression.
    Returns a `ContextualCompressionRetriever` so you can call `.invoke(q)`.
    """

    # --- base hybrid retriever ---
    retriever = PineconeHybridSearchRetriever(
        embeddings=cfg.embeddings,                 # dense
        sparse_encoder=_load_bm25_encoder(),       # sparse (BM25)
        index=_get_pinecone_index(),               # Pinecone Index object
        namespace=cfg.PINECONE_NAMESPACE,
        alpha=alpha,
        top_k=top_k,
        # text_key=text_key,
    )

    # --- Cohere reranker as compressor ---
    compressor = CohereRerank(model="rerank-v3.5", 
                              top_n= 3)

    hybrid_rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever) 
    
    docs = hybrid_rerank_retriever.invoke(query)
    out = []
    for d in docs:
        score = d.metadata.get("relevance_score") 
        out.append({
            "score": score,
            "content": d.page_content,
            # "source": d.metadata.get("source") or d.metadata.get("url"),
        })
    return out


################################################################################
from langchain_neo4j import Neo4jGraph
# ============================================
# Neo4j Knowledge Graph retriever
# ============================================

# helpers
def _get_neo4j_graph() -> Neo4jGraph:
    req = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [k for k in req if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing Neo4j env vars: {', '.join(missing)}")
    return Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        refresh_schema=False,
    )

def _ensure_fulltext_exists(graph: Neo4jGraph, index_name: str = "entity_fulltext") -> None:
    rows = graph.query(
        "SHOW INDEXES YIELD name, type, state WHERE name = $n RETURN name, type, state",
        params={"n": index_name},
    )
    if not rows or rows[0].get("state") != "ONLINE":
        raise RuntimeError(f"Fulltext index '{index_name}' missing or not ONLINE. Run ingest to create it.")

def _normalize(q: str) -> str:
    q = q.lower().replace("?", " ")
    terms = [t for t in q.split() if len(t) > 2]
    return " ".join(terms) or q

# Enf of helpers code
##############################################################################

@tool
def fetch_facts_for_question(query: str, 
                             top_nodes: int = 3, 
                             max_facts: int = 3) -> List[Dict[str, Any]]:

    '''Use this tool for specific questions about entities and their relationships.'''

    graph = _get_neo4j_graph()
    _ensure_fulltext_exists(graph, "entity_fulltext")

    cypher = """
    CALL db.index.fulltext.queryNodes('entity_fulltext', $q)
    YIELD node, score
    WITH node, score
    ORDER BY score DESC
    LIMIT $top_nodes

    OPTIONAL MATCH (node)-[r]-(n2)
    WITH node, score, r, n2
    WHERE r IS NULL
       OR (
            type(r) <> 'MENTIONS'
        AND (n2 IS NULL OR NOT any(l IN labels(n2) WHERE l IN ['Chunk','Paragraph','Page','Span']))
       )

    WITH
      coalesce(node.name, node.id, node.title, node.value) AS subject,
      CASE WHEN r IS NULL THEN null ELSE type(r) END AS rel,
      CASE WHEN n2 IS NULL THEN null ELSE coalesce(n2.name, n2.id, n2.title, n2.value) END AS object,
      coalesce(r.source, node.source, CASE WHEN n2 IS NULL THEN null ELSE n2.source END) AS source,
      score
    RETURN subject, rel, object, source, score
    ORDER BY score DESC
    LIMIT $max_facts
    """

    q1 = _normalize(query)
    rows = graph.query(cypher, params={"q": q1, "top_nodes": int(top_nodes), "max_facts": int(max_facts)})

    if not rows:
        # fallback: OR-joined terms for Lucene
        terms = list({t for t in q1.split() if len(t) > 2})
        if terms:
            q2 = " OR ".join(terms)
            rows = graph.query(cypher, params={"q": q2, "top_nodes": int(top_nodes), "max_facts": int(max_facts)})

    return rows

# ======================================================
# Quick smoke test from CLI (optional)
# ======================================================
# if __name__ == "__main__":
#     q = 'Do you know anything about Discaya?'

#     pinecone_facts = build_pinecone_retriever(q)
#     print(pinecone_facts)

    # for i in pinecone_facts:
        # print(i)


    # graph_facts = fetch_facts_for_question.invoke({"query": q, "k": 2})
    # for i in graph_facts:
    #     print (i)


    


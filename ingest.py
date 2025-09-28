# ingest.py
# 1. get_pinecone_index()
# 2. create_and_upsert_vectors

# 3. upsert_knowledge_graph_from_chunks
# 4. ensure_fulltext_auto

# Note: 
# data_process.py → producer: parse_single_pdf_to_md → load_markdown_to_documents → chunk_documents
# ingest.py → consumer: get_pinecone_index, create_and_upsert_vectors
# api/main.py (FastAPI) → glue: call producer → pass chunks to consumer → return counts

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import config
import hashlib
from tqdm import tqdm
from typing import List
from langchain_core.documents import Document
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION, embeddings


# Note: pure only Pinecone stuff. The orchestration is in FastAPI

# import to fast api
def get_pinecone_index():
    """Initializes and returns a Pinecone index."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing_indexes = [ix["name"] for ix in pc.list_indexes()]
    
    if config.PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {config.PINECONE_INDEX_NAME}")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=1024,  # text-embedding-3-large dimension
            metric="dotproduct",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    return pc.Index(config.PINECONE_INDEX_NAME)


# import to fast api
def create_and_upsert_vectors(index, chunks: List[Document]):
    """Creates dense and sparse vectors and upserts them to Pinecone."""

    texts = [d.page_content for d in chunks]
    dense_vecs = embeddings.embed_documents(texts)
    print("Dense embeddings created...")   

    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump(str(config.BM25_PATH)) # Save for later use
    sparse_vecs = bm25.encode_documents(texts)
    print("BM25 encoder created...")

    # Prepare vectors for upsert
    to_upsert = []
    for doc, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        doc_id = hashlib.sha256(f"{doc.metadata.get('source', '')}|{doc.page_content}".encode("utf-8")).hexdigest()[:32]
        metadata = {**doc.metadata, "context": doc.page_content}
        to_upsert.append({
            "id": doc_id,
            "values": dense,
            "sparse_values": sparse,
            "metadata": metadata,
        })
    
    # Batch upsert to Pinecone
    print(f"Upserting {len(to_upsert)} vectors to Pinecone...")
    batch_size = 32
    for i in tqdm(range(0, len(to_upsert), batch_size)):
        batch = to_upsert[i:i+batch_size]
        index.upsert(vectors=batch, namespace=config.PINECONE_NAMESPACE)
    print("Upsert complete.")

################################################################################################
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

# ingest.py (append)
import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer


# Helper
def ensure_fulltext_auto(graph: Neo4jGraph,
                         index_name: str = "entity_fulltext",
                         props = ("name","id","title","value","md","description")) -> Dict[str, Any]:
    rows = graph.query("""
    WITH $props AS props
    UNWIND props AS p
    MATCH (n)
    WHERE n[p] IS NOT NULL
    WITH collect(DISTINCT labels(n)) AS labelsets
    UNWIND labelsets AS ls
    UNWIND ls AS label
    RETURN DISTINCT label
    ORDER BY label
    """, params={"props": list(props)})

    labels = [r["label"] for r in rows]
    if not labels:
        return {"index": index_name, "labels": [], "props": list(props), "note": "No labels had target props"}

    # Recreate to ensure coverage of all detected labels
    graph.query(f"DROP INDEX {index_name} IF EXISTS;")
    label_union = "|".join(f"`{l}`" for l in labels)
    prop_list   = ",".join(f"n.{p}" for p in props)
    graph.query(f"""
    CREATE FULLTEXT INDEX {index_name}
    FOR (n:{label_union})
    ON EACH [{prop_list}];
    """)
    return {"index": index_name, "labels": labels, "props": list(props)}


# import to fast api
def upsert_knowledge_graph_from_chunks(
    chunks: List[Document],
    *,
    llm,                               # pass a LangChain LLM (e.g., ChatOpenAI)
    fulltext_index_name: str = "entity_fulltext",
    base_entity_label: str = "Entity",
) -> Dict[str, Any]:
    """Turn text chunks into a KG in Neo4j. No-op if NEO4J_* env vars are missing."""
    required = ["NEO4J_URI","NEO4J_USERNAME","NEO4J_PASSWORD"]
    if not all(os.getenv(k) for k in required):
        return {"enabled": False, "reason": "NEO4J_* env vars not set"}

    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        refresh_schema=False,
    )

    # Minimal constraint so MERGE behaves like upsert
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;")

    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = transformer.convert_to_graph_documents(chunks)

    # Store with explicit source metadata and a base label
    graph.add_graph_documents(
        graph_documents,
        include_source=True,
        baseEntityLabel=base_entity_label
    )

    ft = ensure_fulltext_auto(graph, index_name=fulltext_index_name)

    # best-effort counts (GraphDocument has nodes/relationships)
    try:
        node_count = sum(len(gd.nodes) for gd in graph_documents)
        rel_count  = sum(len(gd.relationships) for gd in graph_documents)
    except Exception:
        node_count = rel_count = None

    return {"enabled": True, "nodes": node_count, "rels": rel_count, **ft}




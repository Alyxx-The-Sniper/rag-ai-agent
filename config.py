import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatDeepInfra
from langchain_community.embeddings import DeepInfraEmbeddings
# Generative LLM for the agent
llm_gen = ChatDeepInfra(
    model=os.getenv('GEN_MODEL'),
    temperature=0,
    max_tokens=1024,
    deepinfra_api_key=os.getenv('DEEPINFRA_API_KEY'))
# Embedding model for creating dense vectors
embeddings = DeepInfraEmbeddings(
    model_id=os.getenv('EMBED_MODEL'),
    deepinfra_api_token=os.getenv('DEEPINFRA_API_KEY'),
    normalize=True,)  # Cosine similarity friendly


# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent # pointed to E:\0_AI_2025\railway\rag_082025
BM25_PATH = BASE_DIR / "bm25_values.json"

DATA_DIR = BASE_DIR / "sample_documents" / "PDFs"
TMP_DIR  = BASE_DIR / "sample_documents" / "PDFs_parsed"

# Create directories only after the variables exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


# --- PINECONE CONFIG ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "rag-hybrid-agentic"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = "docs"


# --- NEO4J CONFIG ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
USE_KNOWLEDGE_GRAPH = all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD])

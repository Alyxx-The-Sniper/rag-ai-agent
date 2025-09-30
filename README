# RAG AI Agent Demo

## API Endpoints

1. **PDF Ingestion**
   - Ingestion is password-protected for security and to prevent abuse.
   - I use Marker (a parsing technique) for ingestion. It consumes too much RAM to run inside the web service, so I’ve disabled it.
   - One way to avoid Marker’s high RAM usage is to create a separate ingest service that is invoked only when needed.
   - Another option is to ingest offline (pre-load into Pinecone and Neo4j). This is the approach used in this demo.

2. **Query**
   - Users ask questions about the pre-ingested PDF.
# data_process.py
# 1. marker_parse (helper to parse_folder_to_md)
# 2. parse_single_pdf_to_md/parse_folder_to_md
# 3. load_markdown_to_documents
# 4. chunk_documents

##############################
# Using Marker
##############################
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from typing import List, Dict, Optional
from pathlib import Path

###################################################################################
# initialize once
_PDF_CONVERTER = PdfConverter(artifact_dict=create_model_dict())

# helper 
def marker_parse(pdf_path: str) -> str:
    rendered = _PDF_CONVERTER(pdf_path)
    md, meta, images = text_from_rendered(rendered)
    return md
####################################################################################    
def parse_single_pdf_to_md(pdf_path: str | Path, #  must be either a string (str) or a pathlib.Path object pointing to the PDF file to process.
                           out_dir: str | Path) -> [Dict[str, str]]:
    
    pdf_p = Path(pdf_path) # Convert the incoming pdf_path (might have been a plain string) into a pathlib.Path object so we can use convenient path methods later.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = marker_parse(str(pdf_p))

    md_p = out_dir / f"{pdf_p.stem}.md"
    md_p.write_text(md, encoding="utf-8")
    
    return [{"source_pdf": str(pdf_p),
             "markdown_path": str(md_p)}]

###################################################################################
# ============================================
# Build Documents 
# ============================================
from langchain_core.documents import Document
def load_markdown_to_documents(parsed_data: List[Dict[str, str]]) -> List[Document]:
    """Loads parsed Markdown files into LangChain Document objects."""
    docs = []
    for item in parsed_data:
        with open(item["markdown_path"], "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(
            page_content=text,
            metadata={"source": item["source_pdf"], "md": item["markdown_path"]}
        ))
    return docs

###################################################################################
# ============================================
# Chunking
# ============================================
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Splits a list of Documents into smaller chunks."""
    encoding = tiktoken.get_encoding("cl100k_base")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding.name,
        chunk_size=400,
        chunk_overlap=0,
        separators=[
            "\n```",    # Fenced code blocks
            "\n# ",     # H1
            "\n## ",    # H2
            "\n### ",   # H3
            "\n#### ",  # H4
            "\n\n",     # Paragraphs
            "\n- ", "\n* ", "\n1. ", # Lists
            "\n|",      # Table rows
            "\n",       # Line breaks
            ". ",       # Sentences
            " ", ""
        ],
    )
    
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks




# data_process.py
# 1. marker_parse (helper to parse_folder_to_md)
# 2. parse_single_pdf_to_md/parse_folder_to_md
# 3. load_markdown_to_documents
# 4. chunk_documents

##############################
# Using Marker
##############################
from typing import List, Dict, Optional, TYPE_CHECKING
from pathlib import Path

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    _MARKER_AVAILABLE = True
except Exception:
    # Marker isn't installed in the web image (expected for the demo)
    PdfConverter = None            # type: ignore[assignment]
    create_model_dict = None       # type: ignore[assignment]
    text_from_rendered = None      # type: ignore[assignment]
    _MARKER_AVAILABLE = False

def _require_marker() -> None:
    if not _MARKER_AVAILABLE:
        # Clear error message for callers (UI/API)
        raise ImportError(
            "marker-pdf is not installed in this deployment. "
            "Run locally with: pip install .[ingest]"
        )




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
    
    _require_marker()
    pdf_p = Path(pdf_path) # Convert the incoming pdf_path (might have been a plain string) into a pathlib.Path object so we can use convenient path methods later.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = marker_parse(str(pdf_p))

    md_p = out_dir / f"{pdf_p.stem}.md"
    md_p.write_text(md, encoding="utf-8")
    
    return [{"source_pdf": str(pdf_p),
             "markdown_path": str(md_p)}]

###################################################################################
# def parse_folder_to_md(input_dir: str, out_dir: str) -> List[Dict[str, str]]:
#     out = []
#     for pdf in glob.glob(str(pathlib.Path(input_dir) / "**/*.pdf"), recursive=True):
#         md = marker_parse(pdf)
#         stem = pathlib.Path(pdf).stem
#         md_path = str(pathlib.Path(out_dir) / f"{stem}.md")
#         with open(md_path, "w", encoding="utf-8") as f:
#             f.write(md)
#         out.append({"source_pdf": pdf, "markdown_path": md_path})
#     return out

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




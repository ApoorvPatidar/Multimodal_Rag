"""PDF processing using LangChain's PyPDFLoader for text extraction with metadata.

Includes text cleaning utilities to improve downstream chunking and embeddings
quality (e.g., fix hyphenation, normalize whitespace, and common ligatures).
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
from Utils.logger import logging
import re
import unicodedata


def load_pdf(pdf_path: str) -> List[Document]:
    """Load PDF using LangChain's PyPDFLoader.
    
    Returns:
        List of Document objects with page_content and metadata (page number, source).
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logging.info(f"loaded pdf pages={len(documents)} path={pdf_path}")
        return documents
    except Exception as e:
        logging.exception(f"pdf load failed path={pdf_path}")
        raise


def _clean_text(text: str) -> str:
    """Clean raw PDF-extracted text for better chunking/embeddings.

    - Remove soft hyphenations across line breaks (e.g., "trans-")
    - Normalize different newline styles to spaces while preserving paragraphs
    - Collapse excessive whitespace
    - Normalize common PDF ligatures (ﬁ → fi, ﬂ → fl)
    - Strip control characters
    """
    if not text:
        return ""

    # Normalize unicode (NFKC helps with some width/compatibility chars)
    t = unicodedata.normalize("NFKC", text)

    # Fix common ligatures
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("ﬃ", "ffi").replace("ﬄ", "ffl")

    # Remove hyphenation at line breaks: "trans-\nformer" -> "transformer"
    t = re.sub(r"-\s*\n", "", t)

    # Replace newlines within paragraphs to spaces, but keep paragraph breaks
    # Convert multiple newlines to a placeholder paragraph break, then flatten single newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{2,}", "<PARA>", t)  # mark paragraphs
    t = t.replace("\n", " ")
    t = t.replace("<PARA>", "\n\n")

    # Collapse excessive whitespace
    t = re.sub(r"\s+", " ", t)

    return t.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """Return new Documents with cleaned page_content and preserved metadata.

    Logs total length before/after to help diagnose poor extraction quality.
    """
    before = sum(len(d.page_content or "") for d in documents)
    cleaned: List[Document] = []
    for d in documents:
        cleaned.append(Document(page_content=_clean_text(d.page_content or ""), metadata=d.metadata.copy()))
    after = sum(len(d.page_content or "") for d in cleaned)
    logging.info(f"cleaned pdf text total_chars_before={before} total_chars_after={after}")
    return cleaned


def convert_pdf_to_markdown(path: str) -> str:
    """Legacy compatibility: extract all text as single string.
    
    For backward compatibility with existing code. New code should use load_pdf().
    """
    documents = load_pdf(path)
    documents = clean_documents(documents)
    return "\n\n".join(doc.page_content for doc in documents)


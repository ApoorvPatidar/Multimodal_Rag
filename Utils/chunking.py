"""Text chunking using LangChain's RecursiveCharacterTextSplitter."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from Utils.logger import logging


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks using LangChain's splitter.
    
    Args:
        documents: List of LangChain Document objects.
        chunk_size: Target size for each chunk.
        chunk_overlap: Number of characters to overlap between chunks.
    
    Returns:
        List of Document objects with chunked content and preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logging.info(f"split documents={len(documents)} into chunks={len(chunks)} size={chunk_size} overlap={chunk_overlap}")
    return chunks


def chunk_markdown(markdown_text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Legacy compatibility: chunk plain text string.
    
    For backward compatibility. New code should use chunk_documents().
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(markdown_text)
    logging.info(f"chunked text into chunks={len(chunks)} size={chunk_size} overlap={chunk_overlap}")
    return chunks


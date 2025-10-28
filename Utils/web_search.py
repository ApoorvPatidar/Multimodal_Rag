"""Web search utilities using DuckDuckGo to retrieve context as LangChain Documents."""
from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from langchain.schema import Document
from Utils.logger import logging


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def search_to_documents(query: str, max_results: int = 5) -> List[Document]:
    """Run a web search and convert results to LangChain Documents.

    Each Document.page_content contains a short summary/snippet; metadata includes
    title, url, and source (domain).
    """
    docs: List[Document] = []
    try:
        with DDGS(timeout=10) as ddg:
            results = ddg.text(query, max_results=max_results)
        for r in results or []:
            title = r.get("title") or ""
            href = r.get("href") or r.get("url") or ""
            body = r.get("body") or ""
            content = (title + ": " if title else "") + body
            meta = {
                "title": title,
                "url": href,
                "source": _domain(href),
            }
            docs.append(Document(page_content=content, metadata=meta))
    except Exception as e:
        logging.warning(f"web search failed: {e}")
    logging.info(f"web search query='{query}' docs={len(docs)}")
    return docs

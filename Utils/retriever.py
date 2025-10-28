"""Retriever using LangChain's retriever interface."""

from langchain.schema import Document
from typing import List
from Utils.logger import logging


class Retriever:
    """Wrapper for LangChain retriever with legacy compatibility."""
    
    def __init__(self, vector_store=None, embeddata=None, langchain_retriever=None, *, search_type: str = "similarity", search_kwargs: dict | None = None):
        """Initialize retriever.
        
        Args:
            vector_store: FAISSVectorStore or legacy store.
            embeddata: Legacy EmbedData (deprecated).
            langchain_retriever: Direct LangChain retriever instance.
            search_type: Retrieval mode, e.g., 'similarity' or 'mmr'.
            search_kwargs: Extra kwargs for retriever (e.g., {"k": 6, "fetch_k": 20}).
        """
        self.vector_store = vector_store
        self.embeddata = embeddata
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {"k": 4}
        
        # Prefer native LangChain retriever if provided
        if langchain_retriever:
            self.retriever = langchain_retriever
        elif hasattr(vector_store, 'as_retriever'):
            try:
                # Newer LangChain FAISS retriever supports search_type
                self.retriever = vector_store.as_retriever(search_kwargs=self.search_kwargs)
                # Try to set search_type if supported
                if hasattr(self.retriever, 'search_type'):
                    self.retriever.search_type = self.search_type
            except TypeError:
                # Fallback if search_type not supported in this version
                self.retriever = vector_store.as_retriever(search_kwargs=self.search_kwargs)
        else:
            self.retriever = None
            logging.warning("no LangChain retriever available, using legacy mode")
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for query.
        
        Args:
            query: Query string.
            k: Number of documents to retrieve.
            
        Returns:
            List of Document objects.
        """
        if self.retriever:
            # Use LangChain retriever
            try:
                results = self.retriever.get_relevant_documents(query)
                logging.debug(f"retrieved docs={len(results)} query_len={len(query)}")
                return results
            except Exception as e:
                logging.exception("retrieval failed")
                raise
        else:
            # Legacy fallback
            if hasattr(self.vector_store, 'similarity_search'):
                return self.vector_store.similarity_search(query, k=k)
            elif hasattr(self.vector_store, 'search') and self.embeddata:
                # Old Qdrant stub
                query_embedding = self.embeddata.embed_query(query) if hasattr(self.embeddata, 'embed_query') else []
                results = self.vector_store.search(query_embedding, top_k=k)
                # Convert tuples to Documents
                return [Document(page_content=text, metadata={}) for text, _ in results]
            else:
                raise ValueError("No valid retrieval method available")


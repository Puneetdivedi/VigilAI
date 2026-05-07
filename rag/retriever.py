"""
Retriever for querying the FAISS index.
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

index_dir = os.path.join(os.path.dirname(__file__), "faiss_index")
_vectorstore = None

def get_retriever():
    """Returns the FAISS retriever, loading it if necessary."""
    global _vectorstore
    if _vectorstore is None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            _vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
            return None
    return _vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve_context(query: str) -> str:
    """Retrieves top-k context chunks for a given query."""
    retriever = get_retriever()
    if not retriever:
        return "No maintenance manual context available."
    
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

def is_index_ready() -> bool:
    """Checks if the FAISS index files exist on disk."""
    return os.path.exists(os.path.join(index_dir, "index.faiss"))

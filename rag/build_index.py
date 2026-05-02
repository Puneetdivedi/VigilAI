"""
Embeds the knowledge base and saves a FAISS index.
"""
import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge_base.maintenance_docs import maintenance_chunks

def build_index():
    """Builds and saves the FAISS index."""
    print("Initializing embeddings model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Embedding {len(maintenance_chunks)} chunks...")
    vectorstore = FAISS.from_texts(maintenance_chunks, embeddings)
    
    save_dir = os.path.join(os.path.dirname(__file__), "faiss_index")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    vectorstore.save_local(save_dir)
    print(f"FAISS index saved to {save_dir}")

if __name__ == "__main__":
    build_index()

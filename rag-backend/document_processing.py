from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Add TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib
from config import (get_logger, CHUNK_SIZE, CHUNK_OVERLAP)
import os

logger = get_logger(__name__)

def load_and_split_document(file_path: str) -> List[Document]:
    """Loads and splits a document using LangChain components."""
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(".txt"):
             loader = TextLoader(file_path, encoding="utf-8") 
        else:
            logger.warning(f"Unsupported file type: {file_path}. Skipping.")
            return []

        documents = loader.load() # Returns a list of docs (often 1 per page for PDF)

        if not documents:
             logger.warning(f"No content loaded from {file_path}")
             return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True, 
        )
        split_docs = text_splitter.split_documents(documents)

      
        doc_base_name = os.path.basename(file_path)
        for i, doc in enumerate(split_docs):
            

            doc_content = doc.page_content[:100]  
            content_hash = hashlib.md5(f"{doc_base_name}_{i}_{doc_content}".encode('utf-8')).hexdigest()[:12]
            chunk_id = f"chunk_{content_hash}"
            doc.metadata["id"] = chunk_id
            doc.metadata["chunk_index"] = i
            doc.metadata["source_document"] = doc_base_name
            
            doc.metadata["source"] = doc.metadata.get("source", doc_base_name)
            doc.metadata["page"] = doc.metadata.get("page", None)



        logger.info(f"Loaded and split {file_path} into {len(split_docs)} chunks.")
        return split_docs

    except Exception as e:
        logger.error(f"Failed to load/split document {file_path}: {e}")
        return []
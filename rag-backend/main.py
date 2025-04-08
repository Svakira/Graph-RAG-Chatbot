import time
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from embedder import InfermaticEmbeddings 
from fusion_retriever import get_fusion_retriever

from config import (
    get_logger, 
    UPLOAD_DIR, 
    TOP_K_INITIAL_SEARCH,
    HF_MODEL_NAME,
    INGEST_ENABLE_INTRA_DOC_SIMILARITY,
    INGEST_SIMILARITY_THRESHOLD,
    INGEST_SIMILAR_NEIGHBORS_TO_LINK,
    ENTITY_LABELS_TO_EXTRACT
    )

from model_client import CustomChatQwen
from document_processing import load_and_split_document
from retriever import get_graph_enhanced_retriever, format_docs
from langchain.vectorstores import Chroma
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification # Added imports
logger = get_logger(__name__)


try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_NAME)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple" 
    )
    embedding_model = InfermaticEmbeddings() 
    chat_model = CustomChatQwen()

    chroma_store = Chroma(
        collection_name="rag_embeddings",
        embedding_function=embedding_model,
        persist_directory=os.path.join(UPLOAD_DIR, "chroma_db")
    )
    logger.info("Initialized LangChain components (Embeddings, ChatModel, Neo4jGraph, Neo4jVector)")
except Exception as e:
    logger.exception(f"Fatal error during initialization: {e}")
    raise RuntimeError(f"Failed to initialize core components: {e}")



app = FastAPI(title="LangChain GraphRAG Agent", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_entities(text: str) -> Dict[str, Dict[str, Any]]:
    """Extracts named entities using Hugging Face transformers pipeline."""
    entities = {}
    allowed_labels = set(ENTITY_LABELS_TO_EXTRACT) if ENTITY_LABELS_TO_EXTRACT else None

    try:
        
        ner_results = ner_pipeline(text)
        
        for ent in ner_results:
            entity_label = ent['entity_group']
            # The 'word' field contains the extracted entity text after aggregation
            entity_text = ent['word'].strip()

            # Skip if label is not in our allowed list (if list exists)
            if allowed_labels and entity_label not in allowed_labels:
                continue

            
            if not entity_text or len(entity_text) < 2: 
                 continue

            
            entity_key = entity_text.lower()

            # Store the entity if it's new, or potentially update if score is higher
            if entity_key not in entities or float(ent['score']) > entities[entity_key].get('score', 0.0):
                 entities[entity_key] = {
                     "label": entity_label,
                     "text": entity_text, 
                     "score": float(ent['score']) 
                 }

    except Exception as e:
        
        logger.error(f"Error during Hugging Face NER processing for text segment: {e}")
        
        return {} 

    return entities



def ingest_pipeline(file_path: str, document_id: str):
    """ingestion pipeline with entity extraction (HF) and optimized similarity linking."""
    start_time = time.time()
    try:
        logger.info(f"[Ingest:{document_id}] Starting for: {file_path}")
        split_docs = load_and_split_document(file_path)
        if not split_docs:
            logger.error(f"[Ingest:{document_id}] No documents generated. Aborting.")
            return

        chunk_count = len(split_docs)
        logger.info(f"[Ingest:{document_id}] Split into {chunk_count} chunks.")

        # --- Step 1: Add Chunks to Vector Store ---
        chunk_ids = [d.metadata["id"] for d in split_docs]
        added_ids = chroma_store.add_documents(split_docs, ids=chunk_ids)
        logger.info(f"[Ingest:{document_id}] Added {len(added_ids)} chunks to Chroma vector store.")
        if not added_ids:
             logger.warning(f"[Ingest:{document_id}] No chunks added. Aborting.")
             return

        docs_to_process = [doc for doc in split_docs if doc.metadata["id"] in added_ids]

        # --- Completion ---
        end_time = time.time()
        logger.info(f"[Ingest:{document_id}] Successfully completed ingestion for: {file_path} in {end_time - start_time:.2f} seconds")

    except Exception as e:
        logger.exception(f"[Ingest:{document_id}] Critical error during ingestion pipeline for {file_path}: {e}")


# --- API Endpoints ---
@app.post("/upload-context")
async def upload_context(background_tasks: BackgroundTasks, file: UploadFile):
    sanitized_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)
    document_id = "doc_" + sanitized_filename.replace(".", "_").replace(" ", "_")

    # Save file temporarily
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file {sanitized_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
         if hasattr(file, 'file') and hasattr(file.file, 'close'):
             file.file.close()

    # --- Schedule ingestion as background task ---
    background_tasks.add_task(ingest_pipeline, file_path, document_id)

    return {
        "status": "processing",
        "message": f"File '{sanitized_filename}' received and scheduled for ingestion.",
        "document_id": document_id
    }


@app.post("/chat")
async def chat(message: str = Form(...)):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(f"Received chat message (truncated): {message[:100]}...")

    try:
        fusion_retriever = get_fusion_retriever(chroma_store, chat_model, TOP_K_INITIAL_SEARCH)

        template = """You are a helpful AI assistant. Answer the user's question based *only* 
        on the provided context. If the context does not contain the answer, 
        state that you cannot answer based on the information available. 
        Do not make information up. Be concise and accurate.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": fusion_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | chat_model
            | StrOutputParser()
        )

        reply = await rag_chain.ainvoke(message)

        logger.info("RAG chain finished, returning reply.")
        return {"reply": reply}

    except Exception as e:
        logger.exception(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
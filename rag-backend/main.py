import time
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

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
from embedder import AzureEmbeddings
from model_client import CustomChatQwen
from graph_db import get_neo4j_graph_instance, get_neo4j_vector_store
from document_processing import load_and_split_document
from retriever import get_graph_enhanced_retriever, format_docs
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
    embedding_model = AzureEmbeddings()
    chat_model = CustomChatQwen()
    neo4j_graph = get_neo4j_graph_instance() 
    neo4j_vector_store = get_neo4j_vector_store(embedding_model)
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
        added_ids = neo4j_vector_store.add_documents(split_docs, ids=chunk_ids)
        logger.info(f"[Ingest:{document_id}] Added {len(added_ids)} chunks to vector store.")
        if not added_ids:
             logger.warning(f"[Ingest:{document_id}] No chunks added. Aborting.")
             return

        # --- Step 2: Create Document Node and CONTAINS Relationships ---
        logger.info(f"[Ingest:{document_id}] Linking chunks to Document node...")
        neo4j_graph.query("""
            MERGE (d:Document {id: $doc_id})
            SET d.filename = $filename, d.last_updated = timestamp()
            WITH d
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {id: chunk_id})
            MERGE (d)-[:CONTAINS]->(c)
        """, params={"doc_id": document_id, "filename": os.path.basename(file_path), "chunk_ids": added_ids})

        # --- Step 3: Create NEXT_CHUNK Relationships ---
        logger.info(f"[Ingest:{document_id}] Adding NEXT_CHUNK relationships...")
        sorted_chunk_ids = [d.metadata["id"] for d in sorted(split_docs, key=lambda d: d.metadata["chunk_index"]) if d.metadata["id"] in added_ids]
        batch_params = [{"prev_id": sorted_chunk_ids[i], "curr_id": sorted_chunk_ids[i+1]}
                        for i in range(len(sorted_chunk_ids) - 1)]
        if batch_params:
            neo4j_graph.query("""
                UNWIND $batch AS pair
                MATCH (prev:Chunk {id: pair.prev_id})
                MATCH (curr:Chunk {id: pair.curr_id})
                MERGE (prev)-[:NEXT_CHUNK]->(curr)
            """, params={"batch": batch_params})
            logger.info(f"[Ingest:{document_id}] Added {len(batch_params)} NEXT_CHUNK relationships.")

        # --- Step 4: Extract Entities (HF) and Create MENTIONS Relationships ---
        logger.info(f"[Ingest:{document_id}] Extracting entities (HF) and linking MENTIONS...")
        entity_links_batch = []
        processed_entities = set() # Track unique entity keys (lowercase names)

        docs_to_process = [doc for doc in split_docs if doc.metadata["id"] in added_ids]

        
        for doc in docs_to_process:
            chunk_id = doc.metadata["id"]
            # *** Calls the new Hugging Face based function ***
            entities = extract_entities(doc.page_content)

            for entity_key, entity_data in entities.items(): 
                 
                 if entity_key not in processed_entities:
                      entity_links_batch.append({
                          "chunk_id": None, # Flag for entity merge
                          "entity_name": entity_key, 
                          "entity_label": entity_data["label"],
                          "entity_text": entity_data["text"] 
                      })
                      processed_entities.add(entity_key)
                 
                 entity_links_batch.append({
                     "chunk_id": chunk_id,
                     "entity_name": entity_key, 
                     "entity_label": None,
                     "entity_text": None
                 })

        if entity_links_batch:
             # Cypher query 
             neo4j_graph.query("""
                UNWIND [item IN $batch WHERE item.chunk_id IS NULL] AS entity_data
                MERGE (e:Entity {name: entity_data.entity_name}) // Use lowercase 'name' as unique key
                ON CREATE SET e.label = entity_data.entity_label, e.text = entity_data.entity_text, e.created = timestamp()
                ON MATCH SET e.last_seen = timestamp(), e.label = coalesce(e.label, entity_data.entity_label), e.text = coalesce(e.text, entity_data.entity_text) // Update label/text if missing

                WITH $batch AS batch_data
                UNWIND [item IN batch_data WHERE item.chunk_id IS NOT NULL] AS link_data
                MATCH (c:Chunk {id: link_data.chunk_id})
                MATCH (e:Entity {name: link_data.entity_name}) // Match entity using lowercase name
                MERGE (c)-[:MENTIONS]->(e)
             """, params={"batch": entity_links_batch})
             logger.info(f"[Ingest:{document_id}] Processed {len(processed_entities)} unique entities (HF) and created MENTIONS links.")

        # --- Step 5: Create SIMILAR_TO Relationships using Vector Index ---
        
        logger.info(f"[Ingest:{document_id}] Creating SIMILAR_TO relationships (threshold > {INGEST_SIMILARITY_THRESHOLD})...")
        
        cypher_query_similar = f"""
            MATCH (new:Chunk) WHERE new.id IN $chunk_ids
            CALL db.index.vector.queryNodes('chunk_embeddings', $k_similar, new.embedding) YIELD node AS similar_candidate, score
            WITH new, similar_candidate, score
            WHERE score > $threshold AND new <> similar_candidate
            {'''
            WITH new, similar_candidate, score
            MATCH (new)<-[:CONTAINS]-(d1:Document)
            MATCH (similar_candidate)<-[:CONTAINS]-(d2:Document)
            WHERE d1 <> d2
            ''' if not INGEST_ENABLE_INTRA_DOC_SIMILARITY else ''}
            RETURN new.id as id1, similar_candidate.id as id2, score
        """
        try:
            results = neo4j_graph.query(cypher_query_similar, params={
                "chunk_ids": added_ids,
                "k_similar": INGEST_SIMILAR_NEIGHBORS_TO_LINK * 2, 
                "threshold": INGEST_SIMILARITY_THRESHOLD
            })
            similar_links_batch = []
            processed_pairs = set()
            for record in results:
                id1, id2, score = record["id1"], record["id2"], record["score"]
                pair = tuple(sorted((id1, id2)))
                if pair not in processed_pairs:
                    similar_links_batch.append({"id1": id1, "id2": id2, "score": score})
                    processed_pairs.add(pair)

            if similar_links_batch:
                neo4j_graph.query("""
                    UNWIND $batch AS link
                    MATCH (c1:Chunk {id: link.id1})
                    MATCH (c2:Chunk {id: link.id2})
                    MERGE (c1)-[r:SIMILAR_TO]->(c2)
                    SET r.score = link.score
                """, params={"batch": similar_links_batch})
                logger.info(f"[Ingest:{document_id}] Added {len(similar_links_batch)} SIMILAR_TO relationships.")
            else:
                 logger.info(f"[Ingest:{document_id}] No new SIMILAR_TO relationships met the threshold.")
        except Exception as e:
             logger.error(f"[Ingest:{document_id}] Failed during SIMILAR_TO linking using vector index: {e}")

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
        # --- Define RAG Chain using LCEL ---
        vector_retriever = neo4j_vector_store.as_retriever(search_kwargs={'k': TOP_K_INITIAL_SEARCH})
        graph_enhanced_retriever = get_graph_enhanced_retriever(vector_retriever, neo4j_graph)

        # System Prompt
        template = """You are a helpful AI assistant. Answer the user's question based *only* 
        on the provided context. If the context does not contain the answer, s
        tate that you cannot answer based on the information available. 
        Do not make information up. Be concise and accurate.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        
        rag_chain = (
            # RunnableParallel allows fetching context and passing question through simultaneously
            {"context": graph_enhanced_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | chat_model # Use the custom chat model wrapper
            | StrOutputParser() # Parse the output message content
        )

        # --- Invoke the RAG chain ---
        logger.info("Invoking RAG chain...")
        reply = await rag_chain.ainvoke(message) #  async invoke for FastAPI
       

        logger.info("RAG chain finished, returning reply.")
        return {"reply": reply}

    except ValueError as e:
        
        logger.error(f"ValueError during chat processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        logger.error(f"LLM ConnectionError during chat processing: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to connect to language model: {e}")
    except TimeoutError as e:
        logger.error(f"LLM TimeoutError during chat processing: {e}")
        raise HTTPException(status_code=504, detail=f"Language model request timed out: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
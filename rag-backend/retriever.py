from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from config import (
    GRAPH_CONTEXT_NEIGHBORS,
    get_logger)

logger = get_logger(__name__)
def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string for the LLM context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_graph_enhanced_retriever(vector_retriever: BaseRetriever, graph: Neo4jGraph, k_neighbors: int = GRAPH_CONTEXT_NEIGHBORS) -> BaseRetriever:
    """
    A retriever that combines vector search with graph neighborhood traversal.
    """
    def fetch_neighbors(docs: List[Document]) -> List[Document]:
        """Fetches neighboring chunks from the graph for the initially retrieved docs."""
        if not docs or k_neighbors <= 0:
            return docs #

        all_docs_map = {doc.metadata["id"]: doc for doc in docs}
        neighbor_query = """
        MATCH (c:Chunk) WHERE c.id IN $chunk_ids
        
        CALL {
            // Get Top K Sequential Previous
            MATCH (c:Chunk) WHERE c.id IN $chunk_ids
            MATCH path = (prev:Chunk)-[:NEXT_CHUNK*1..2]->(c)
            RETURN prev as neighbor, length(path) as dist, 'sequential_prev' as rel_type
            ORDER BY dist DESC
            LIMIT $k_neighbors
        UNION
            // Get Top K Sequential Next
            MATCH (c:Chunk) WHERE c.id IN $chunk_ids
            MATCH path = (c)-[:NEXT_CHUNK*1..2]->(next:Chunk)
            RETURN next as neighbor, length(path) as dist, 'sequential_next' as rel_type
            ORDER BY dist ASC
            LIMIT $k_neighbors
        UNION
            // Get Top K Semantic Neighbors
            MATCH (c:Chunk) WHERE c.id IN $chunk_ids
            MATCH (c)-[sim:SIMILAR_TO]-(similar:Chunk) // Undirected
            WHERE sim.score > 0.75 AND similar <> c   // Use your desired threshold
            RETURN similar as neighbor, sim.score as dist, 'semantic' as rel_type
            ORDER BY dist DESC
            LIMIT $k_neighbors
        }
        
        // Filter out chunks we already have
        WITH neighbor, dist, rel_type
        WHERE NOT neighbor.id IN $chunk_ids

        // Return the distinct neighbors that passed the filter
        RETURN DISTINCT
            neighbor.id AS id,
            neighbor.text AS text,
            neighbor.source_document AS source_document,
            neighbor.chunk_index AS chunk_index,
            rel_type AS relationship_type,
            dist AS relationship_weight
        """
        
        chunk_ids = list(all_docs_map.keys())
        try:
            results = graph.query(neighbor_query, params={
                "chunk_ids": chunk_ids, 
                "k_neighbors": k_neighbors
            })
            
           
            sequential_neighbors = 0
            semantic_neighbors = 0
            
            for record in results:
                neighbor_id = record["id"]
                if neighbor_id not in all_docs_map:
                    neighbor_doc = Document(
                        page_content=record["text"],
                        metadata={
                            "id": neighbor_id,
                            "source_document": record["source_document"],
                            "chunk_index": record["chunk_index"],
                            "retrieval_source": record["relationship_type"],
                            "relationship_weight": record["relationship_weight"]
                        }
                    )
                    all_docs_map[neighbor_id] = neighbor_doc
                    
                    
                    if record["relationship_type"].startswith("sequential"):
                        sequential_neighbors += 1
                    elif record["relationship_type"] == "semantic":
                        semantic_neighbors += 1
            
            total_neighbors = sequential_neighbors + semantic_neighbors
            logger.info(f"Fetched {total_neighbors} unique neighbor chunks: "
                    f"{sequential_neighbors} sequential and {semantic_neighbors} semantic")
        except Exception as e:
            logger.error(f"Failed to fetch graph neighbors: {e}")
            

       
        return list(all_docs_map.values())

    
    graph_retriever_runnable = (
        vector_retriever # Input: query string -> Output: List[Document]
        | RunnableLambda(fetch_neighbors) # Input: List[Document] -> Output: List[Document] (original + neighbors)
    )
    return graph_retriever_runnable

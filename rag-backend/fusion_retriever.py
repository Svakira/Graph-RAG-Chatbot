from langchain.retrievers.multi_query import MultiQueryRetriever
from config import get_logger

logger = get_logger(__name__)

class LoggedMultiQueryRetriever(MultiQueryRetriever):
    def generate_queries(self, question: str):
        queries = super().generate_queries(question)
        logger.info(f"Fusion RAG generated queries for '{question}': {queries}")
        return queries

def get_fusion_retriever(chroma_store, llm, top_k=5):
    retriever = LoggedMultiQueryRetriever.from_llm(
        retriever=chroma_store.as_retriever(search_kwargs={"k": top_k}),
        llm=llm
    )
    return retriever

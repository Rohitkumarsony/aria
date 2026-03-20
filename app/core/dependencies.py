from functools import lru_cache
from app.agents.rag_agent import RAGAgent
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from app.validators.response_validator import ResponseValidator


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore: return VectorStore()

@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    r = HybridRetriever(get_vector_store()); r.rebuild(); return r

@lru_cache(maxsize=1)
def get_agent() -> RAGAgent: return RAGAgent(get_retriever())

@lru_cache(maxsize=1)
def get_validator() -> ResponseValidator: return ResponseValidator()

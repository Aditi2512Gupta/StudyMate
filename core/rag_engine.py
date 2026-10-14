"""
RAG Engine Module for StudyMate.
Coordinates the end-to-end Retrieval-Augmented Generation pipeline:
Query Embedding -> FAISS Vector Search -> Context Retrieval & Formatting -> Grounded LLM Generation.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from core import prompts
from core import gemini_client
from core.vector_store import FAISSVectorStore
import config


def answer_query_with_rag(
    query: str,
    vector_store: FAISSVectorStore,
    api_key: Optional[str] = None,
    top_k: int = config.DEFAULT_TOP_K,
    model: str = config.DEFAULT_LLM_MODEL
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes the complete RAG pipeline for a user question.
    
    Args:
        query: User question string.
        vector_store: Initialized FAISSVectorStore containing document chunks.
        api_key: Gemini API key.
        top_k: Number of relevant context chunks to retrieve.
        model: Gemini model identifier.
        
    Returns:
        Tuple of (generated_answer: str, retrieved_chunks: List[Dict[str, Any]])
    """
    if vector_store.is_empty():
        return (
            "⚠️ No document has been processed yet. Please upload a PDF study material in the sidebar "
            "or load the sample study guide first.",
            []
        )

    # 1. Embed query
    effective_key = config.get_api_key(api_key)
    if effective_key:
        query_vec = gemini_client.generate_embeddings([query], api_key=effective_key)
    else:
        # Fallback offline embedding
        query_vec = gemini_client.generate_offline_fallback_embeddings([query], dimension=vector_store.dimension)

    # 2. Retrieve Top-K most relevant chunks using FAISS Cosine Similarity
    retrieved_chunks = vector_store.search(query_vec, top_k=top_k)

    if not retrieved_chunks:
        return (
            "I could not find any relevant passages in the uploaded document that relate to your question.",
            []
        )

    # 3. Format retrieved context with page citations
    context_blocks = []
    for c in retrieved_chunks:
        sim_pct = int(c.get("similarity_score", 0.0) * 100)
        header = f"[Source: Page {c['page_number']}, Chunk #{c['chunk_id']} | Relevance: {sim_pct}%]"
        context_blocks.append(f"{header}\n{c['text']}")

    context_str = "\n\n".join(context_blocks)

    # 4. Synthesize prompt using engineered RAG prompt template
    prompt = prompts.RAG_QA_PROMPT_TEMPLATE.format(
        context_chunks=context_str,
        question=query
    )

    # 5. Generate grounded response using Gemini
    if effective_key:
        answer = gemini_client.generate_text(
            prompt=prompt,
            api_key=effective_key,
            model=model,
            temperature=config.TEMPERATURE_QA
        )
    else:
        # Informative offline preview response showing retrieved chunks
        answer = (
            "ℹ️ **Offline Preview Mode (API Key not configured):**\n\n"
            f"FAISS successfully retrieved the **{len(retrieved_chunks)} most relevant passages** "
            "from your uploaded document for this question.\n\n"
            "To generate live natural language answers synthesized by Gemini LLM, "
            "please enter your **Gemini API Key** in the sidebar."
        )

    return answer, retrieved_chunks

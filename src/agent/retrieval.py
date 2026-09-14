"""Stage 5: retrieval that compresses before it injects.

Naive RAG stuffs the top-20 chunks into the prompt. Most are near-duplicates or
only topically related. ``ContextualCompressionRetriever`` runs a cheap filter
over candidates and forwards only the sentences that actually matter.
"""

from __future__ import annotations

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter

from .config import BUDGET


def build_compressed_retriever(base_retriever, embeddings, llm=None):
    """Wrap a vector retriever in a split -> dedupe -> relevance -> extract chain.

    Order matters: the cheap embedding-based stages cut the candidate set first,
    so the expensive LLM extractor only ever sees a handful of chunks.
    """
    base_retriever.search_kwargs = {
        **getattr(base_retriever, "search_kwargs", {}),
        "k": BUDGET.top_k_before_rerank,
    }

    stages = [
        CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator=". "),
        EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=0.95),
        EmbeddingsFilter(embeddings=embeddings, k=BUDGET.top_k_after_rerank),
    ]
    if llm is not None:
        stages.append(LLMChainExtractor.from_llm(llm))

    return ContextualCompressionRetriever(
        base_compressor=DocumentCompressorPipeline(transformers=stages),
        base_retriever=base_retriever,
    )

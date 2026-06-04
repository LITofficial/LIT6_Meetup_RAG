import os
from typing import Any

from schemas import PipelineStage, QueryResponse


def build_query_response(
    query: str,
    top_k: int,
    documents: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    vector_store: dict[str, Any],
    retrieval_candidates: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    prompt_context: dict[str, Any],
    answer: str,
    model_name: str,
    use_gemini: bool,
) -> QueryResponse:
    stages = build_pipeline_stages(
        query=query,
        top_k=top_k,
        documents=documents,
        chunks=chunks,
        vector_store=vector_store,
        retrieval_candidates=retrieval_candidates,
        matches=matches,
        answer=answer,
        model_name=model_name,
        use_gemini=use_gemini,
        prompt_preview=prompt_context,
    )
    return QueryResponse(
        answer=answer,
        stages=stages,
        sources=build_sources(matches),
        model=model_name,
    )


def build_pipeline_stages(
    query: str,
    top_k: int,
    documents: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    vector_store: dict[str, Any],
    retrieval_candidates: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    answer: str,
    model_name: str,
    use_gemini: bool,
    prompt_preview: dict[str, Any],
) -> list[PipelineStage]:
    vector_items = vector_store["items"]
    return [
        build_chunk_stage(documents, chunks),
        build_embedding_stage(chunks, vector_items, use_gemini),
        build_vector_store_stage(vector_items, vector_store),
        build_retrieval_stage(query, top_k, retrieval_candidates, matches),
        build_context_stage(matches, prompt_preview),
        build_generation_stage(answer, model_name, prompt_preview),
    ]


def build_chunk_stage(
    documents: list[dict[str, str]],
    chunks: list[dict[str, Any]],
) -> PipelineStage:
    return PipelineStage(
        id="chunk",
        title="청킹",
        input=[
            {
                "title": document["title"],
                "filename": document["filename"],
                "text": document["text"][:1200],
                "chars": len(document["text"]),
            }
            for document in documents
        ],
        output=[{"id": chunk["id"], "text": chunk["text"]} for chunk in chunks],
        metric=chunk_metric(chunks),
    )


def build_embedding_stage(
    chunks: list[dict[str, Any]],
    vector_items: list[dict[str, Any]],
    use_gemini: bool,
) -> PipelineStage:
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001") if use_gemini else "local-demo"
    return PipelineStage(
        id="embed",
        title="임베딩",
        input=[chunk["id"] for chunk in chunks],
        output=[
            {"id": item["id"], "embedding_preview": item["embedding"][:4]}
            for item in vector_items
        ],
        metric=embedding_model,
    )


def build_vector_store_stage(
    vector_items: list[dict[str, Any]],
    vector_store: dict[str, Any],
) -> PipelineStage:
    return PipelineStage(
        id="store",
        title="벡터 저장",
        input=[{"chunk_id": item["id"], "metadata": item["filename"]} for item in vector_items],
        output={"collection": vector_store.get("collection", "demo_documents"), "index": vector_store["index"]},
        metric=f"{len(vector_items)} vectors",
    )


def build_retrieval_stage(
    query: str,
    top_k: int,
    retrieval_candidates: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> PipelineStage:
    return PipelineStage(
        id="retrieve",
        title="검색",
        input={"query": query, "top_k": top_k},
        output={
            "candidates": [
                {
                    "chunk_id": candidate["id"],
                    "score": candidate["score"],
                    "text": candidate["text"],
                }
                for candidate in retrieval_candidates
            ],
            "top_k": [
                {
                    "chunk_id": match["id"],
                    "score": match["score"],
                    "text": match["text"],
                }
                for match in matches
            ],
        },
        metric=f"top-{top_k}",
    )


def build_context_stage(
    matches: list[dict[str, Any]],
    prompt_preview: dict[str, Any],
) -> PipelineStage:
    return PipelineStage(
        id="augment",
        title="컨텍스트 조립",
        input=[match["id"] for match in matches],
        output=prompt_preview,
        metric=f"{sum(len(match['tokens']) for match in matches)} tokens",
    )


def build_generation_stage(
    answer: str,
    model_name: str,
    prompt_preview: dict[str, Any],
) -> PipelineStage:
    return PipelineStage(
        id="generate",
        title="답변 생성",
        input=prompt_preview,
        output={"answer": answer, "model": model_name},
        metric=model_name,
    )


def chunk_metric(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "0 chunks"

    return (
        f"{len(chunks)} chunks"
        f" · size {chunks[0].get('chunk_size')}"
        f" · overlap {chunks[0].get('chunk_overlap')}"
    )


def build_sources(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": match["id"],
            "title": match["title"],
            "filename": match["filename"],
            "score": match["score"],
            "text": match["text"],
        }
        for match in matches
    ]

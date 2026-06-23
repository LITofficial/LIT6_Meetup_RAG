import os
from typing import Any

import numpy as np

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
    points = pca_3d_points(vector_items)
    points_by_id = {item["id"]: points[index] for index, item in enumerate(vector_items)}
    return [
        build_chunk_stage(documents, chunks),
        build_embedding_stage(chunks, vector_items, use_gemini, points),
        build_vector_store_stage(vector_items, vector_store),
        build_retrieval_stage(query, top_k, retrieval_candidates, matches, points_by_id),
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
    points: list[dict[str, float]],
) -> PipelineStage:
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001") if use_gemini else "local-demo"
    return PipelineStage(
        id="embed",
        title="임베딩",
        input=[chunk["id"] for chunk in chunks],
        output=[
            {
                "id": item["id"],
                "embedding_preview": item["embedding"][:4],
                "point3d": points[index],
            }
            for index, item in enumerate(vector_items)
        ],
        metric=embedding_model,
    )


def pca_3d_points(vector_items: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not vector_items:
        return []

    vectors = np.array([item.get("full_embedding") or item["embedding"] for item in vector_items], dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        return [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in vector_items]

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, singular_values, components_t = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(3, components_t.shape[0])
    projected = centered @ components_t[:dimensions].T

    if dimensions < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - dimensions)))

    scales = np.percentile(np.abs(projected), 95, axis=0)
    fallback_scales = np.max(np.abs(projected), axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, fallback_scales)
    scales = np.where(scales > 0, scales, 1.0)
    normalized = np.clip((projected / scales) * 28, -36, 36)

    return [
        {
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
            "z": round(float(point[2]), 4),
            "weight": round(float(singular_values[0] if len(singular_values) else 0), 4),
        }
        for point in normalized
    ]


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
    points_by_id: dict[str, dict[str, float]],
) -> PipelineStage:
    query_point = weighted_query_point(matches, points_by_id)
    return PipelineStage(
        id="retrieve",
        title="검색",
        input={"query": query, "top_k": top_k, "query_point3d": query_point},
        output={
            "candidates": [
                {
                    "chunk_id": candidate["id"],
                    "score": candidate["score"],
                    "text": candidate["text"],
                    "point3d": points_by_id.get(candidate["id"]),
                }
                for candidate in retrieval_candidates
            ],
            "top_k": [
                {
                    "chunk_id": match["id"],
                    "score": match["score"],
                    "text": match["text"],
                    "point3d": points_by_id.get(match["id"]),
                }
                for match in matches
            ],
            "query_point3d": query_point,
        },
        metric=f"top-{top_k}",
    )


def weighted_query_point(matches: list[dict[str, Any]], points_by_id: dict[str, dict[str, float]]) -> dict[str, float]:
    weighted_points = []
    for match in matches:
        point = points_by_id.get(match["id"])
        if not point:
            continue
        weighted_points.append((max(float(match.get("score") or 0), 0.01), point))

    if not weighted_points:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    total = sum(weight for weight, _ in weighted_points)
    return {
        axis: round(sum(weight * point[axis] for weight, point in weighted_points) / total, 4)
        for axis in ("x", "y", "z")
    }


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

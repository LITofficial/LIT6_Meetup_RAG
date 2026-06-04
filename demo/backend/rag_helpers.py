import asyncio
import hashlib
import math
import os
import re
import time
from io import BytesIO
from typing import Any

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_config import PDF_TEXT_CACHE, QUERY_EMBEDDING_CACHE, STOPWORDS, logger


async def read_uploaded_file(file: UploadFile) -> tuple[bytes, str, str, str, str | None, float]:
    started_at = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    filename = file.filename or "uploaded.pdf"
    content_type = file.content_type or ""
    digest = hashlib.sha1(contents).hexdigest()[:10]
    return contents, filename, content_type, digest, PDF_TEXT_CACHE.get(digest), started_at


async def document_from_upload(file: UploadFile) -> dict[str, Any]:
    contents, filename, content_type, digest, cached_text, started_at = await read_uploaded_file(file)
    text = extract_text_from_file(contents, filename, content_type, digest, cached_text)
    text = normalize_extracted_text(text)
    return uploaded_document_payload(filename, digest, text, cached_text, contents, started_at)


def extract_text_from_file(
    contents: bytes,
    filename: str,
    content_type: str,
    digest: str,
    cached_text_value: str | None,
) -> str:
    if cached_text_value is not None:
        return cached_text_value

    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(BytesIO(contents))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        PDF_TEXT_CACHE[digest] = text
        logger.info("upload.extract_pdf filename=%s pages=%s", filename, len(reader.pages))
        return text

    return contents.decode("utf-8", errors="ignore")


def normalize_extracted_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="파일에서 읽을 수 있는 텍스트를 찾지 못했습니다.")
    return text


def uploaded_document_payload(
    filename: str,
    digest: str,
    text: str,
    cached_text_value: str | None,
    contents: bytes,
    started_at: float,
) -> dict[str, Any]:
    logger.info(
        "upload.loaded filename=%s bytes=%s chars=%s cached=%s seconds=%.2f",
        filename,
        len(contents),
        len(text),
        cached_text_value is not None,
        time.perf_counter() - started_at,
    )
    return {
        "id": f"upload_{digest}",
        "filename": filename,
        "title": filename.rsplit(".", 1)[0],
        "text": text,
        "cached_text": cached_text_value is not None,
    }


def split_documents_adaptively(
    splitter: RecursiveCharacterTextSplitter,
    documents: list[Document],
) -> tuple[list[Document], int, int]:
    target_chunks = int(os.getenv("RAG_TARGET_CHUNKS", "180"))
    max_chunk_size = int(os.getenv("RAG_MAX_CHUNK_SIZE", "2400"))
    chunk_size = splitter._chunk_size
    chunk_overlap = splitter._chunk_overlap
    separators = splitter._separators

    while True:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(chunk_overlap, max(0, chunk_size // 3)),
            separators=separators,
        )
        split_docs = splitter.split_documents(documents)

        if len(split_docs) <= target_chunks or chunk_size >= max_chunk_size:
            logger.info(
                "chunking.done chunks=%s chunk_size=%s chunk_overlap=%s target_chunks=%s",
                len(split_docs),
                chunk_size,
                min(chunk_overlap, max(0, chunk_size // 3)),
                target_chunks,
            )
            return split_docs, chunk_size, min(chunk_overlap, max(0, chunk_size // 3))

        next_chunk_size = min(max_chunk_size, int(chunk_size * 1.25))
        logger.info(
            "chunking.resize chunks=%s target_chunks=%s chunk_size=%s next_chunk_size=%s",
            len(split_docs),
            target_chunks,
            chunk_size,
            next_chunk_size,
        )
        chunk_size = next_chunk_size


def chunk_payloads(
    split_docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    chunks = []

    for split_doc in split_docs:
        document_id = split_doc.metadata["document_id"]
        counters[document_id] = counters.get(document_id, 0) + 1
        chunks.append(
            {
                "id": f"{document_id}_chunk_{counters[document_id]:03d}",
                "document_id": document_id,
                "filename": split_doc.metadata["filename"],
                "title": split_doc.metadata["title"],
                "text": split_doc.page_content,
                "tokens": tokenize(split_doc.page_content),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

    return chunks


async def embed_documents_in_batches(
    embeddings: GoogleGenerativeAIEmbeddings,
    texts: list[str],
) -> list[list[float]]:
    batch_size = int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", "24"))
    max_retries = int(os.getenv("GEMINI_EMBEDDING_MAX_RETRIES", "3"))
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_number = (start // batch_size) + 1
        total_batches = math.ceil(len(texts) / batch_size)

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "vector_store.embedding_batch_start batch=%s/%s size=%s attempt=%s",
                    batch_number,
                    total_batches,
                    len(batch),
                    attempt + 1,
                )
                vectors.extend(await embeddings.aembed_documents(batch))
                break
            except GoogleGenerativeAIError as error:
                if not is_resource_exhausted_error(error) or attempt >= max_retries:
                    raise

                wait_seconds = min(20, (2 ** attempt) * 3)
                logger.warning(
                    "vector_store.embedding_rate_limited batch=%s/%s wait_seconds=%s error=%s",
                    batch_number,
                    total_batches,
                    wait_seconds,
                    error,
                )
                await asyncio.sleep(wait_seconds)

    return vectors


def is_resource_exhausted_error(error: Exception) -> bool:
    detail = str(error)
    return "RESOURCE_EXHAUSTED" in detail or "429" in detail


def build_local_vector_store(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": [{**chunk, "embedding": local_demo_embedding(chunk["text"])} for chunk in chunks],
        "chroma": None,
        "index": "in-memory cosine",
    }


def local_chunk_embeddings(chunks: list[dict[str, Any]], use_gemini: bool) -> list[list[float]] | None:
    if use_gemini:
        return None
    return [local_demo_embedding(chunk["text"]) for chunk in chunks]


def reusable_vector_store(
    documents: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    use_gemini: bool,
    cache: dict[str, Any],
) -> dict[str, Any] | None:
    cached = get_cached_vector_store(documents, use_gemini, cache)
    if cached:
        return cached
    if not chunks:
        return {"items": [], "chroma": None, "index": "empty"}
    if not use_gemini:
        return cache_vector_store(documents, chunks, use_gemini, build_local_vector_store(chunks), cache)
    return None


def get_cached_vector_store(documents: list[dict[str, str]], use_gemini: bool, cache: dict[str, Any]) -> dict[str, Any] | None:
    cached = cache.get(documents_cache_key(documents, use_gemini))
    return cached[1] if cached else None


def cache_vector_store(
    documents: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    use_gemini: bool,
    vector_store: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    cache[documents_cache_key(documents, use_gemini)] = (chunks, vector_store)
    return vector_store


def build_collection_name(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha1("".join(chunk["id"] for chunk in chunks).encode("utf-8")).hexdigest()[:16]
    return f"rag_cosine_{digest}"


def format_vector_items(
    chunks: list[dict[str, Any]],
    chunk_embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    return [
        {
            **chunk,
            "embedding": [round(value, 4) for value in embedding[:12]],
            "full_embedding": embedding,
        }
        for chunk, embedding in zip(chunks, chunk_embeddings, strict=False)
    ]


async def embed_chunks_with_retry(
    embeddings: GoogleGenerativeAIEmbeddings,
    chunks: list[dict[str, Any]],
) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    try:
        return await embeddings.aembed_documents(texts)
    except GoogleGenerativeAIError as error:
        if is_resource_exhausted_error(error):
            return await embed_documents_in_batches(embeddings, texts)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini 임베딩 생성에 실패했습니다. GEMINI_EMBEDDING_MODEL 값을 확인하세요. details={error}",
        ) from error


def vector_store_payload(
    chunks: list[dict[str, Any]],
    chunk_embeddings: list[list[float]],
    chroma: Any,
    collection_name: str,
) -> dict[str, Any]:
    return {
        "items": format_vector_items(chunks, chunk_embeddings),
        "chroma": chroma,
        "index": "Chroma cosine",
        "collection": collection_name,
    }


def format_chroma_results(results: list[tuple[Document, float]]) -> list[dict[str, Any]]:
    candidates = []

    for document, distance in results:
        metadata = document.metadata
        similarity = max(0.0, 1.0 - float(distance))
        candidates.append(
            {
                "id": metadata["id"],
                "document_id": metadata["document_id"],
                "filename": metadata["filename"],
                "title": metadata["title"],
                "text": document.page_content,
                "tokens": tokenize(document.page_content),
                "score": round(similarity, 4),
            }
        )

    return candidates


def upsert_chunks_to_chroma(
    chroma: Any,
    chunks: list[dict[str, Any]],
    chunk_embeddings: list[list[float]],
    collection_name: str,
) -> None:
    upsert_started_at = time.perf_counter()
    chroma._collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=chunk_embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "id": chunk["id"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "title": chunk["title"],
            }
            for chunk in chunks
        ],
    )
    logger.info(
        "vector_store.chroma_upsert_done collection=%s seconds=%.2f",
        collection_name,
        time.perf_counter() - upsert_started_at,
    )


async def embed_query_for_local_search(query: str, use_gemini: bool) -> list[float]:
    if not use_gemini:
        return local_demo_embedding(query)

    cache_key = query_embedding_cache_key(query)
    query_embedding = QUERY_EMBEDDING_CACHE.get(cache_key)
    if query_embedding is not None:
        return query_embedding

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=require_google_api_key(),
    )
    query_embedding = await embeddings.aembed_query(query)
    QUERY_EMBEDDING_CACHE[cache_key] = query_embedding
    return query_embedding


def rank_chunks_by_cosine_similarity(
    query_embedding: list[float],
    vector_items: list[dict[str, Any]],
    candidate_k: int,
) -> list[dict[str, Any]]:
    scored = []
    for item in vector_items:
        item_embedding = item.get("full_embedding") or item["embedding"]
        score = cosine_similarity(query_embedding, item_embedding)
        scored.append({**item, "score": round(score, 4)})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:candidate_k]


def build_context(matches: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"근거 {index + 1}\n"
        f"- 파일: {match['filename']}\n"
        f"- 제목: {match['title']}\n"
        f"- 내용: {match['text']}"
        for index, match in enumerate(matches)
    )


def prompt_context_payload(query: str, matches: list[dict[str, Any]], context: str, prompt_preview: str) -> dict[str, Any]:
    return {
        "system": "LangChain ChatPromptTemplate: 검색된 근거만 사용해 답변하고 출처를 포함합니다.",
        "query": query,
        "contexts": [match["text"] for match in matches],
        "context": context,
        "prompt_preview": prompt_preview,
    }


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def require_google_api_key() -> str:
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY가 설정되지 않았습니다. backend 환경변수에 Gemini API key를 추가하세요.",
        )
    if not api_key.isascii():
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY가 올바른 Gemini API key 형식이 아닙니다.",
        )
    return api_key


def documents_cache_key(documents: list[dict[str, str]], use_gemini: bool) -> str:
    digest = hashlib.sha1()
    digest.update(os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001").encode("utf-8"))
    digest.update(str(use_gemini).encode("utf-8"))
    for document in documents:
        digest.update(document["id"].encode("utf-8"))
        digest.update(document["filename"].encode("utf-8"))
        digest.update(document["text"].encode("utf-8"))
    return digest.hexdigest()


def query_embedding_cache_key(query: str) -> str:
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    return hashlib.sha1(f"{model_name}\n{query}".encode("utf-8")).hexdigest()


def answer_cache_key(query: str, matches: list[dict[str, Any]], model_name: str) -> str:
    digest = hashlib.sha1()
    digest.update(model_name.encode("utf-8"))
    digest.update(query.encode("utf-8"))
    for match in matches:
        digest.update(match["id"].encode("utf-8"))
        digest.update(match["text"].encode("utf-8"))
    return digest.hexdigest()


def cached_answer(query: str, matches: list[dict[str, Any]], model_name: str, cache: dict[str, Any]) -> tuple[str, str] | None:
    return cache.get(answer_cache_key(query, matches, model_name))


def cache_answer(
    query: str,
    matches: list[dict[str, Any]],
    model_name: str,
    answer: str,
    cache: dict[str, Any],
) -> tuple[str, str]:
    payload = (answer, model_name)
    cache[answer_cache_key(query, matches, model_name)] = payload
    return payload


def local_demo_embedding(text: str, dimensions: int = 12) -> list[float]:
    vector = [0.0] * dimensions

    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1 if digest[2] % 2 == 0 else -1
        vector[index] += sign

    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / magnitude, 4) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def clean_answer(answer: str) -> str:
    cleaned = re.sub(r"\[\d+\]", "", answer)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def compose_fallback_answer(query: str, matches: list[dict[str, Any]]) -> str:
    primary = matches[0] if matches else None
    if not primary:
        return "관련 문서 근거를 찾지 못했습니다."

    evidence = " ".join(match["text"] for match in matches)
    if "환불" in query:
        return (
            f"업로드된 문서 기준으로는 다음 근거가 가장 관련 있습니다. {evidence} "
            f"가장 높은 유사도 근거는 {primary['filename']}의 '{primary['title']}'입니다."
        )

    return f"검색된 문서 근거를 기준으로 답변하면, {evidence}"

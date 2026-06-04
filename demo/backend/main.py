import asyncio
import hashlib
import logging
import math
import os
import re
import time
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pydantic import BaseModel, Field


logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="RAG Pipeline Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SAMPLE_DOCUMENTS = [
    {
        "id": "policy_refund",
        "filename": "sample_policy.pdf",
        "title": "신규 고객 환불 정책",
        "text": (
            "신규 고객은 결제일로부터 14일 이내에 환불을 요청할 수 있습니다. "
            "단, 서비스 사용량이 전체 제공량의 30%를 초과하면 부분 환불만 가능합니다. "
            "환불 요청은 고객 지원 채널에서 주문 번호와 함께 접수해야 합니다."
        ),
    },
    {
        "id": "shipping_exception",
        "filename": "operations_note.md",
        "title": "배송 예외 처리",
        "text": (
            "배송 지연이 7일을 초과하면 고객에게 보상 쿠폰을 제공합니다. "
            "천재지변이나 주소 오류로 발생한 지연은 환불 정책과 별도로 검토합니다."
        ),
    },
    {
        "id": "support_faq",
        "filename": "faq.csv",
        "title": "고객 지원 FAQ",
        "text": (
            "고객은 영업일 기준 2일 안에 접수 결과를 안내받습니다. "
            "정책 판단이 필요한 문의는 담당자가 문서 근거를 확인한 뒤 답변합니다."
        ),
    },
]

STOPWORDS = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "와",
    "과",
    "로",
    "으로",
    "및",
    "the",
    "a",
    "an",
}

VECTOR_STORE_CACHE: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
QUERY_EMBEDDING_CACHE: dict[str, list[float]] = {}
ANSWER_CACHE: dict[str, tuple[str, str]] = {}
PDF_TEXT_CACHE: dict[str, str] = {}

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 업로드된 문서를 바탕으로 답변하는 한국어 RAG 어시스턴트다.\n"
            "아래 규칙을 반드시 지켜라.\n"
            "1. 제공된 근거 내용만 사용한다.\n"
            "2. 답변은 자연스러운 한국어 문단으로 작성한다.\n"
            "3. LaTeX 수식 표기, Markdown 굵게 표시, 글머리표, [1] 같은 각주 번호를 쓰지 않는다.\n"
            "4. 수식은 일반 텍스트로 풀어 쓴다. 예: omega_avg = delta theta / delta t\n"
            "5. 출처는 답변 마지막 줄에 '출처: 파일명' 형식으로 한 번만 쓴다.\n"
            "6. 근거가 부족하면 부족하다고 말하고 추측하지 않는다.",
        ),
        (
            "human",
            "사용자 질문:\n{query}\n\n검색된 근거:\n{context}\n\n최종 답변:",
        ),
    ]
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=5)


class PipelineStage(BaseModel):
    id: str
    title: str
    input: Any
    output: Any
    metric: str


class QueryResponse(BaseModel):
    answer: str
    stages: list[PipelineStage]
    sources: list[dict[str, Any]]
    model: str


# 데모에서 보여줄 RAG 파이프라인 함수들
# 순서: PDF 읽기 -> 청킹 -> 임베딩/저장 -> 검색 -> 컨텍스트 조립 -> LLM 답변
# 1단계: PDF 업로드/텍스트 추출
# 발표에서는 "PDF를 모델에 바로 넣는 것이 아니라 먼저 읽을 수 있는 텍스트로 바꾼다"는 부분입니다.
async def document_from_upload(file: UploadFile) -> dict[str, Any]:
    started_at = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    filename = file.filename or "uploaded.pdf"
    content_type = file.content_type or ""
    digest = hashlib.sha1(contents).hexdigest()[:10]
    cached_text_value = PDF_TEXT_CACHE.get(digest)

    if cached_text_value is not None:
        text = cached_text_value
        logger.info("upload.cache_hit filename=%s chars=%s seconds=%.2f", filename, len(text), time.perf_counter() - started_at)
    elif filename.lower().endswith(".pdf") or content_type == "application/pdf":
        extract_started_at = time.perf_counter()
        reader = PdfReader(BytesIO(contents))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        PDF_TEXT_CACHE[digest] = text
        logger.info(
            "upload.extract_pdf filename=%s pages=%s seconds=%.2f",
            filename,
            len(reader.pages),
            time.perf_counter() - extract_started_at,
        )
    else:
        text = contents.decode("utf-8", errors="ignore")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="파일에서 읽을 수 있는 텍스트를 찾지 못했습니다.")

    logger.info(
        "upload.loaded filename=%s bytes=%s chars=%s seconds=%.2f",
        filename,
        len(contents),
        len(text),
        time.perf_counter() - started_at,
    )
    return {
        "id": f"upload_{digest}",
        "filename": filename,
        "title": filename.rsplit(".", 1)[0],
        "text": text,
        "cached_text": cached_text_value is not None,
    }


# 2단계 준비: 추출된 텍스트를 LangChain Document 객체로 변환합니다.
# LangChain의 splitter, retriever, chain 구성요소가 metadata를 함께 다룰 수 있게 만드는 단계입니다.
def build_langchain_documents(documents: list[dict[str, str]]) -> list[Document]:
    return [
        Document(
            page_content=document["text"],
            metadata={
                "document_id": document["id"],
                "filename": document["filename"],
                "title": document["title"],
            },
        )
        for document in documents
    ]


# 2단계: 청킹
# LangChain RecursiveCharacterTextSplitter로 긴 문서를 작은 문단 카드처럼 나눕니다.
def split_documents_with_langchain(documents: list[Document]) -> list[dict[str, Any]]:
    split_docs, chunk_size, chunk_overlap = split_documents_with_adaptive_size(documents)
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


def split_documents_with_adaptive_size(documents: list[Document]) -> tuple[list[Document], int, int]:
    target_chunks = int(os.getenv("RAG_TARGET_CHUNKS", "180"))
    max_chunk_size = int(os.getenv("RAG_MAX_CHUNK_SIZE", "2400"))
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    separators = ["\n\n", "\n", ". ", "다. ", " ", ""]

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


# 3단계: 임베딩 모델 준비
# LangChain GoogleGenerativeAIEmbeddings로 청크와 질문을 같은 벡터 공간에 올립니다.
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=require_google_api_key(),
    )


def is_resource_exhausted_error(error: Exception) -> bool:
    detail = str(error)
    return "RESOURCE_EXHAUSTED" in detail or "429" in detail


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


# 3단계: 임베딩/벡터 저장
# 각 청크를 Gemini 임베딩으로 바꾸고, 검색에 필요한 metadata와 함께 Chroma 벡터 저장소를 만듭니다.
async def build_vector_store(
    chunks: list[dict[str, Any]],
    use_gemini: bool,
) -> dict[str, Any]:
    if not chunks:
        return {"items": [], "chroma": None, "index": "empty"}

    if not use_gemini:
        return {
            "items": [{**chunk, "embedding": local_demo_embedding(chunk["text"])} for chunk in chunks],
            "chroma": None,
            "index": "in-memory cosine",
        }

    embeddings = get_embedding_model()
    collection_name = "rag_cosine_" + hashlib.sha1("".join(chunk["id"] for chunk in chunks).encode("utf-8")).hexdigest()[:16]
    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    try:
        embedding_started_at = time.perf_counter()
        logger.info("vector_store.embedding_start chunks=%s", len(chunks))
        chunk_embeddings = await embed_documents_in_batches(
            embeddings,
            [chunk["text"] for chunk in chunks],
        )
        logger.info(
            "vector_store.embedding_done chunks=%s seconds=%.2f",
            len(chunk_embeddings),
            time.perf_counter() - embedding_started_at,
        )

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
    except GoogleGenerativeAIError as error:
        if is_resource_exhausted_error(error):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini 임베딩 API 사용량 제한에 걸렸습니다. "
                    "잠시 후 다시 실행하거나 업로드 문서 수 또는 chunk 수를 줄이세요. "
                    f"details={error}"
                ),
            ) from error
        raise HTTPException(
            status_code=502,
            detail=f"Gemini 임베딩 생성에 실패했습니다. GEMINI_EMBEDDING_MODEL 값을 확인하세요. details={error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Chroma 벡터 저장소 생성에 실패했습니다. details={error}",
        ) from error
    return {
        "items": [
            {
                **chunk,
                "embedding": [round(value, 4) for value in embedding[:12]],
                "full_embedding": embedding,
            }
            for chunk, embedding in zip(chunks, chunk_embeddings, strict=False)
        ],
        "chroma": chroma,
        "index": "Chroma cosine",
        "collection": collection_name,
    }


async def get_cached_chunks_and_vector_store(
    documents: list[dict[str, str]],
    use_gemini: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_key = documents_cache_key(documents, use_gemini)
    cached = VECTOR_STORE_CACHE.get(cache_key)
    if cached:
        return cached

    langchain_documents = build_langchain_documents(documents)
    chunks = split_documents_with_langchain(langchain_documents)
    vector_store = await build_vector_store(chunks, use_gemini)
    VECTOR_STORE_CACHE[cache_key] = (chunks, vector_store)
    return chunks, vector_store


# 4단계: 검색
# 사용자 질문도 임베딩한 뒤, 벡터 저장소에서 의미적으로 가까운 Top-K 청크를 고릅니다.
async def retrieve(
    query: str,
    top_k: int,
    vector_store: dict[str, Any],
    use_gemini: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_k = min(len(vector_store["items"]), max(top_k * 3, 5))

    if use_gemini and vector_store.get("chroma"):
        results = await vector_store["chroma"].asimilarity_search_with_score(query, k=candidate_k)
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
        return candidates, candidates[:top_k]

    if use_gemini:
        cache_key = query_embedding_cache_key(query)
        query_embedding = QUERY_EMBEDDING_CACHE.get(cache_key)
        if query_embedding is None:
            embeddings = get_embedding_model()
            query_embedding = await embeddings.aembed_query(query)
            QUERY_EMBEDDING_CACHE[cache_key] = query_embedding
    else:
        query_embedding = local_demo_embedding(query)

    scored = []

    for item in vector_store["items"]:
        item_embedding = item.get("full_embedding") or item["embedding"]
        score = cosine_similarity(query_embedding, item_embedding)
        scored.append({**item, "score": round(score, 4)})

    candidates = sorted(scored, key=lambda item: item["score"], reverse=True)
    return candidates[:candidate_k], candidates[:top_k]


# 5단계 준비: 검색된 청크를 Gemini에게 넘길 근거 문단 형태로 정리합니다.
def build_context(matches: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"근거 {index + 1}\n"
        f"- 파일: {match['filename']}\n"
        f"- 제목: {match['title']}\n"
        f"- 내용: {match['text']}"
        for index, match in enumerate(matches)
    )


# 5단계: 프롬프트 조립
# LangChain ChatPromptTemplate에 사용자 질문과 검색된 근거를 넣어 최종 모델 입력을 만듭니다.
def build_prompt(query: str, matches: list[dict[str, Any]]) -> str:
    return RAG_PROMPT.format(query=query, context=build_context(matches))


# 6단계: 답변 생성
# LangChain Runnable 체인(prompt | llm)으로 Gemini를 호출해 근거 기반 답변을 생성합니다.
async def generate_answer_with_gemini(query: str, matches: list[dict[str, Any]]) -> tuple[str, str]:
    api_key = require_google_api_key()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    cache_key = answer_cache_key(query, matches, model_name)
    cached_answer = ANSWER_CACHE.get(cache_key)
    if cached_answer:
        return cached_answer

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
    )
    chain = RAG_PROMPT | llm
    response = await chain.ainvoke({"query": query, "context": build_context(matches)})
    answer = (clean_answer(str(response.content)), model_name)
    ANSWER_CACHE[cache_key] = answer
    return answer


# 전체 RAG 파이프라인
# PDF 텍스트 -> LangChain Document -> 청킹 -> 임베딩 -> 검색 -> 프롬프트 조립 -> Gemini 답변까지 순서대로 실행합니다.
async def run_pipeline(
    query: str,
    top_k: int = 3,
    documents: list[dict[str, str]] | None = None,
    use_gemini: bool = True,
) -> QueryResponse:
    total_started_at = time.perf_counter()
    step_started_at = total_started_at
    documents = documents or SAMPLE_DOCUMENTS
    logger.info(
        "pipeline.start documents=%s query_chars=%s use_gemini=%s",
        len(documents),
        len(query),
        use_gemini,
    )

    chunks, vector_store = await get_cached_chunks_and_vector_store(documents, use_gemini)
    logger.info(
        "pipeline.step chunk_embed_store chunks=%s seconds=%.2f",
        len(chunks),
        time.perf_counter() - step_started_at,
    )

    step_started_at = time.perf_counter()
    vector_items = vector_store["items"]
    retrieval_candidates, matches = await retrieve(query, top_k, vector_store, use_gemini)
    logger.info(
        "pipeline.step retrieve matches=%s seconds=%.2f",
        len(matches),
        time.perf_counter() - step_started_at,
    )

    step_started_at = time.perf_counter()
    if use_gemini:
        answer, model_name = await generate_answer_with_gemini(query, matches)
    else:
        answer = compose_fallback_answer(query, matches)
        model_name = "local-demo"
    logger.info(
        "pipeline.step generate model=%s seconds=%.2f",
        model_name,
        time.perf_counter() - step_started_at,
    )

    prompt_preview = {
        "system": "LangChain ChatPromptTemplate: 검색된 근거만 사용해 답변하고 출처를 포함합니다.",
        "query": query,
        "contexts": [match["text"] for match in matches],
        "prompt_preview": build_prompt(query, matches),
    }
    chunk_metric = (
        f"{len(chunks)} chunks"
        f" · size {chunks[0].get('chunk_size')}"
        f" · overlap {chunks[0].get('chunk_overlap')}"
        if chunks
        else "0 chunks"
    )

    stages = [
        PipelineStage(
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
            metric=chunk_metric,
        ),
        PipelineStage(
            id="embed",
            title="임베딩",
            input=[chunk["id"] for chunk in chunks],
            output=[
                {"id": item["id"], "embedding_preview": item["embedding"][:4]}
                for item in vector_items
            ],
            metric=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001") if use_gemini else "local-demo",
        ),
        PipelineStage(
            id="store",
            title="벡터 저장",
            input=[{"chunk_id": item["id"], "metadata": item["filename"]} for item in vector_items],
            output={"collection": vector_store.get("collection", "demo_documents"), "index": vector_store["index"]},
            metric=f"{len(vector_items)} vectors",
        ),
        PipelineStage(
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
        ),
        PipelineStage(
            id="augment",
            title="컨텍스트 조립",
            input=[match["id"] for match in matches],
            output=prompt_preview,
            metric=f"{sum(len(match['tokens']) for match in matches)} tokens",
        ),
        PipelineStage(
            id="generate",
            title="답변 생성",
            input=prompt_preview,
            output={"answer": answer, "model": model_name},
            metric=model_name,
        ),
    ]

    sources = [
        {
            "chunk_id": match["id"],
            "title": match["title"],
            "filename": match["filename"],
            "score": match["score"],
            "text": match["text"],
        }
        for match in matches
    ]

    response = QueryResponse(answer=answer, stages=stages, sources=sources, model=model_name)
    logger.info("pipeline.done stages=%s total_seconds=%.2f", len(stages), time.perf_counter() - total_started_at)
    return response


# 내부 보조 함수
# 데모 발표에서는 핵심 파이프라인을 먼저 보여주고, 아래 함수들은 필요할 때만 설명합니다.
# 보조 함수: 청크와 질문을 비교하기 전에 불필요한 조사/짧은 단어를 줄입니다.
def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


# 보조 함수: LangChain의 Gemini 모델/임베딩 호출에 필요한 API key를 확인합니다.
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

# 데모 fallback: API key 없이도 흐름을 설명할 수 있게 만든 간단한 로컬 임베딩입니다.
# 실제 발표/데모 품질은 GoogleGenerativeAIEmbeddings 경로를 기준으로 봅니다.
def local_demo_embedding(text: str, dimensions: int = 12) -> list[float]:
    vector = [0.0] * dimensions

    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1 if digest[2] % 2 == 0 else -1
        vector[index] += sign

    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / magnitude, 4) for value in vector]


# 4단계 검색 보조: 질문 벡터와 청크 벡터가 얼마나 같은 방향을 보는지 점수화합니다.
def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


# 보조 함수: 발표 화면에 그대로 보여도 어색하지 않도록 모델 출력의 각주/Markdown 흔적을 정리합니다.
def clean_answer(answer: str) -> str:
    cleaned = re.sub(r"\[\d+\]", "", answer)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# 데모 fallback: Gemini API를 쓰지 않는 경우 검색된 근거를 이어 붙여 흐름만 확인합니다.
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

# 프론트 시각화용: 각 RAG 단계를 WebSocket으로 순서대로 보내 데이터 이동처럼 보여줍니다.
async def stream_pipeline(websocket: WebSocket, query: str, top_k: int) -> None:
    response = await run_pipeline(query=query, top_k=top_k)

    for index, stage in enumerate(response.stages):
        await websocket.send_json(
            {
                "type": "stage",
                "stage": stage.model_dump(),
                "active_stage": index,
                "progress": round(((index + 1) / len(response.stages)) * 100),
            }
        )
        await asyncio.sleep(0.7)

    streamed = ""
    for char in response.answer:
        streamed += char
        await websocket.send_json({"type": "answer_delta", "answer": streamed})
        await asyncio.sleep(0.025)

    await websocket.send_json(
        {
            "type": "completed",
            "answer": response.answer,
            "sources": response.sources,
        }
    )


# 상태 확인용 엔드포인트입니다.
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# 샘플 문서로 전체 RAG 흐름을 빠르게 확인하는 엔드포인트입니다.
@app.get("/api/rag/demo", response_model=QueryResponse)
async def get_demo_pipeline() -> QueryResponse:
    return await run_pipeline(query="신규 고객 환불 정책을 근거와 함께 알려줘")


# PDF 없이 기본 샘플 데이터에 질문만 던지는 엔드포인트입니다.
@app.post("/api/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    return await run_pipeline(query=request.query, top_k=request.top_k)


# 실제 데모용 엔드포인트: 여러 PDF/텍스트 파일과 질문을 받아 하나의 RAG 인덱스로 실행합니다.
@app.post("/api/rag/upload-query", response_model=QueryResponse)
async def query_uploaded_pdf(
    files: list[UploadFile] = File(...),
    query: str = Form(...),
    top_k: int = Form(3),
) -> QueryResponse:
    if not files:
        raise HTTPException(status_code=400, detail="업로드할 파일이 필요합니다.")

    documents = [await document_from_upload(upload_file) for upload_file in files]
    return await run_pipeline(query=query, top_k=top_k, documents=documents)


# 선택 사항: REST 응답 대신 단계별 진행 상황을 실시간으로 보여줄 때 사용하는 WebSocket입니다.
@app.websocket("/ws/rag")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                await websocket.send_json({"type": "error", "message": "query가 필요합니다."})
                continue
            top_k = int(data.get("top_k") or 3)
            await stream_pipeline(websocket, query=query, top_k=top_k)
    except WebSocketDisconnect:
        return

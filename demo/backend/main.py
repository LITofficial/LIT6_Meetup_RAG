import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_helpers as helpers
from rag_config import ANSWER_CACHE, RAG_PROMPT, SAMPLE_DOCUMENTS, VECTOR_STORE_CACHE, logger
from response_builders import build_query_response
from schemas import QueryResponse
from server import create_app


# 전체 RAG 파이프라인
# 준비된 텍스트 문서 -> 청킹 -> 임베딩 -> 벡터 저장 -> 검색 -> 프롬프트 조립 -> Gemini 답변까지 순서대로 실행합니다.
async def run_pipeline(query: str, top_k: int = 3, documents: list[dict[str, str]] | None = None, use_gemini: bool = True) -> QueryResponse:
    documents = documents or SAMPLE_DOCUMENTS
    logger.info("pipeline.start documents=%s query_chars=%s use_gemini=%s", len(documents), len(query), use_gemini)

    chunks = chunk_documents(documents)
    embedding_model = prepare_embedding_model(use_gemini)
    chunk_embeddings = await embed_chunks(documents, chunks, embedding_model, use_gemini)
    vector_store = store_vectors(documents, chunks, chunk_embeddings, embedding_model, use_gemini)
    retrieval_candidates, matches = await retrieve_relevant_chunks(query, top_k, vector_store, use_gemini)
    prompt_context = assemble_prompt_context(query, matches)
    answer, model_name = await generate_grounded_answer(query, matches, prompt_context, use_gemini)

    response = build_query_response(
        query, top_k, documents, chunks, vector_store, retrieval_candidates, matches, prompt_context, answer, model_name, use_gemini
    )
    logger.info("pipeline.done stages=%s", len(response.stages))
    return response


# 1단계: 청킹
# LangChain Document와 RecursiveCharacterTextSplitter로 긴 문서를 검색 가능한 작은 청크로 나눕니다.
def chunk_documents(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    langchain_documents = [
        Document(page_content=document["text"], metadata={"document_id": document["id"], "filename": document["filename"], "title": document["title"]})
        for document in documents
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1200")), chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")), separators=["\n\n", "\n", ". ", "다. ", " ", ""]
    )
    split_docs, chunk_size, chunk_overlap = helpers.split_documents_adaptively(splitter, langchain_documents)
    return helpers.chunk_payloads(split_docs, chunk_size, chunk_overlap)


# 2단계 준비: 임베딩 모델 준비
# Gemini Embeddings를 한 번 준비해 문서 임베딩과 Chroma 검색에 같이 사용합니다.
def prepare_embedding_model(use_gemini: bool) -> GoogleGenerativeAIEmbeddings | None:
    if not use_gemini:
        return None
    return GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"), google_api_key=helpers.require_google_api_key())


# 2단계: 임베딩
# Gemini Embeddings로 청크 텍스트를 의미 벡터로 변환합니다.
async def embed_chunks(
    documents: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    embedding_model: GoogleGenerativeAIEmbeddings | None,
    use_gemini: bool,
) -> list[list[float]] | None:
    if helpers.reusable_vector_store(documents, chunks, use_gemini, VECTOR_STORE_CACHE):
        return None
    local_embeddings = helpers.local_chunk_embeddings(chunks, use_gemini)
    if local_embeddings is not None:
        return local_embeddings
    return await helpers.embed_chunks_with_retry(embedding_model, chunks)


# 3단계: 벡터 저장
# Chroma에 임베딩 벡터와 청크 metadata를 저장합니다.
def store_vectors(
    documents: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    chunk_embeddings: list[list[float]] | None,
    embedding_model: GoogleGenerativeAIEmbeddings | None,
    use_gemini: bool,
) -> dict[str, Any]:
    reusable = helpers.reusable_vector_store(documents, chunks, use_gemini, VECTOR_STORE_CACHE)
    if reusable:
        return reusable
    if chunk_embeddings is None:
        raise ValueError("저장할 임베딩 벡터가 없습니다.")
    collection_name = helpers.build_collection_name(chunks)
    chroma = Chroma(collection_name=collection_name, embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine"})
    helpers.upsert_chunks_to_chroma(chroma, chunks, chunk_embeddings, collection_name)
    vector_store = helpers.vector_store_payload(chunks, chunk_embeddings, chroma, collection_name)
    return helpers.cache_vector_store(documents, chunks, use_gemini, vector_store, VECTOR_STORE_CACHE)


# 4단계: 질문 임베딩/검색
# Chroma에서 사용자 질문과 의미적으로 가까운 Top-K 청크를 찾습니다.
async def retrieve_relevant_chunks(
    query: str,
    top_k: int,
    vector_store: dict[str, Any],
    use_gemini: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_k = min(len(vector_store["items"]), max(top_k * 3, 5))

    if use_gemini and vector_store.get("chroma"):
        results = await vector_store["chroma"].asimilarity_search_with_score(query, k=candidate_k)
        candidates = helpers.format_chroma_results(results)
        return candidates, candidates[:top_k]

    query_embedding = await helpers.embed_query_for_local_search(query, use_gemini)
    candidates = helpers.rank_chunks_by_cosine_similarity(query_embedding, vector_store["items"], candidate_k)
    return candidates, candidates[:top_k]


# 5단계: 컨텍스트/프롬프트 조립
# ChatPromptTemplate에 사용자 질문과 검색된 근거를 넣어 최종 모델 입력을 만듭니다.
def assemble_prompt_context(query: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    context = helpers.build_context(matches)
    prompt_preview = RAG_PROMPT.format(query=query, context=context)
    return helpers.prompt_context_payload(query, matches, context, prompt_preview)


# 6단계: 근거 기반 답변 생성
# LangChain Runnable 체인(prompt | llm)으로 Gemini를 호출합니다.
async def generate_grounded_answer(
    query: str,
    matches: list[dict[str, Any]],
    prompt_context: dict[str, Any],
    use_gemini: bool,
) -> tuple[str, str]:
    if not use_gemini:
        return helpers.compose_fallback_answer(query, matches), "local-demo"

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    answer = helpers.cached_answer(query, matches, model_name, ANSWER_CACHE)
    if answer:
        return answer

    llm = ChatGoogleGenerativeAI(model=model_name, api_key=helpers.require_google_api_key(), temperature=0.2)
    chain = RAG_PROMPT | llm
    response = await chain.ainvoke({"query": query, "context": prompt_context["context"]})
    return helpers.cache_answer(query, matches, model_name, helpers.clean_answer(str(response.content)), ANSWER_CACHE)


app = create_app(run_pipeline=run_pipeline, document_from_upload=helpers.document_from_upload)

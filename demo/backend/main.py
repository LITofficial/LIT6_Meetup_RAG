import hashlib
import os
import unicodedata
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_helpers as helpers
from rag_config import ANSWER_CACHE, RAG_PROMPT, VECTOR_STORE_CACHE, logger
from response_builders import build_chunk_stage, build_query_response
from schemas import QueryResponse
from server import create_app


DEMO_DOCUMENT_CACHE: list[dict[str, str]] | None = None
DEMO_CHUNKS_CACHE: list[dict[str, Any]] | None = None


# 전체 RAG 파이프라인
# 준비된 텍스트 문서 -> 청킹 -> 임베딩 -> 벡터 저장 -> 검색 -> 프롬프트 조립 -> Gemini 답변까지 순서대로 실행합니다.
async def run_pipeline(query: str, top_k: int = 3, documents: list[dict[str, str]] | None = None, use_gemini: bool | None = None) -> QueryResponse:
    uses_demo_document = documents is None
    documents = documents or load_demo_documents()
    use_gemini = resolve_gemini_mode(use_gemini)
    logger.info("pipeline.start documents=%s query_chars=%s use_gemini=%s", len(documents), len(query), use_gemini)

    chunks = load_demo_chunks(documents) if uses_demo_document else chunk_documents(documents)
    embedding_model = prepare_embedding_model(use_gemini)
    chunk_embeddings = await embed_chunks(documents, chunks, embedding_model, use_gemini)
    vector_store = store_vectors(documents, chunks, chunk_embeddings, embedding_model, use_gemini)
    retrieval_candidates, matches = await retrieve_relevant_chunks(query, top_k, vector_store, use_gemini)
    matches = relevant_matches(query, matches)
    prompt_context = assemble_prompt_context(query, matches)
    answer, model_name = await generate_grounded_answer(query, matches, prompt_context, use_gemini)

    response = build_query_response(
        query, top_k, documents, chunks, vector_store, retrieval_candidates, matches, prompt_context, answer, model_name, use_gemini
    )
    logger.info("pipeline.done stages=%s", len(response.stages))
    return response


async def preload_demo_document() -> dict[str, Any]:
    documents = load_demo_documents()
    chunks = load_demo_chunks(documents)
    return {"stages": [build_chunk_stage(documents, chunks)]}


def resolve_gemini_mode(use_gemini: bool | None) -> bool:
    if use_gemini is not None:
        return use_gemini
    return bool((os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'"))


def load_demo_documents() -> list[dict[str, str]]:
    global DEMO_DOCUMENT_CACHE
    if DEMO_DOCUMENT_CACHE is not None:
        return DEMO_DOCUMENT_CACHE

    DEMO_DOCUMENT_CACHE = []
    for document_path in resolve_demo_document_paths():
        document_id = demo_document_id(document_path)
        text_override = demo_text_override(document_path)
        if text_override is None:
            contents = document_path.read_bytes()
            text = helpers.extract_text_from_file(contents, document_path.name, "application/pdf", document_id, None)
        else:
            text = text_override.read_text(encoding="utf-8")
            logger.info("demo_document.text_override filename=%s override=%s", document_path.name, text_override.name)
        text = helpers.normalize_extracted_text(text)
        DEMO_DOCUMENT_CACHE.append(
            {
                "id": document_id,
                "filename": document_path.name,
                "title": document_path.stem,
                "text": text,
            }
        )
        logger.info("demo_document.loaded filename=%s chars=%s", document_path.name, len(text))

    return DEMO_DOCUMENT_CACHE


def load_demo_chunks(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    global DEMO_CHUNKS_CACHE
    if DEMO_CHUNKS_CACHE is None:
        DEMO_CHUNKS_CACHE = chunk_documents(documents)
        logger.info("demo_document.chunked chunks=%s", len(DEMO_CHUNKS_CACHE))
    return DEMO_CHUNKS_CACHE


def resolve_demo_document_paths() -> list[Path]:
    configured_path = (os.getenv("DEMO_DOCUMENT_PATH") or "").strip()
    if configured_path:
        path = Path(configured_path)
        if path.exists():
            return [path]
        raise HTTPException(status_code=500, detail=f"데모 문서를 찾을 수 없습니다: {configured_path}")

    document_dirs = [
        Path(os.getenv("DEMO_DOCUMENT_DIR", "/app/document")),
        Path(__file__).resolve().parent.parent / "document",
    ]

    for document_dir in document_dirs:
        if not document_dir.exists():
            continue
        paths = sorted(document_dir.glob("*.pdf"), key=lambda path: normalize_document_name(path.name))
        if paths:
            return paths

    raise HTTPException(status_code=500, detail="document 폴더에서 데모 PDF를 찾지 못했습니다.")


def normalize_document_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).replace(" ", "").replace("_", "")


def demo_document_id(path: Path) -> str:
    normalized_name = normalize_document_name(path.stem)
    if "경북대학교" in normalized_name and "학칙" in normalized_name:
        return "demo_kyungpook_rules"
    if "컴퓨터학부" in normalized_name and "졸업요건" in normalized_name:
        return "demo_computer_graduation_requirements"
    digest = hashlib.sha1(normalized_name.encode("utf-8")).hexdigest()[:10]
    return f"demo_{digest}"


def demo_text_override(path: Path) -> Path | None:
    normalized_name = normalize_document_name(path.stem)
    if "컴퓨터학부" in normalized_name and "졸업요건" in normalized_name:
        markdown_path = path.with_name("졸업요건.md")
        if markdown_path.exists():
            return markdown_path
    return None


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
    # chromadb 객체
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
    candidate_k = len(vector_store["items"]) if not use_gemini else min(len(vector_store["items"]), max(top_k * 8, 20))

    if use_gemini and vector_store.get("chroma"):
        results = await vector_store["chroma"].asimilarity_search_with_score(query, k=candidate_k)
        candidates = prioritize_keyword_matches(query, helpers.format_chroma_results(results))
        return candidates, candidates[:top_k]

    query_embedding = await helpers.embed_query_for_local_search(query, use_gemini)
    candidates = prioritize_keyword_matches(query, helpers.rank_chunks_by_cosine_similarity(query_embedding, vector_store["items"], candidate_k))
    return candidates, candidates[:top_k]


def prioritize_keyword_matches(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_tokens = set(helpers.tokenize(query))
    if not query_tokens:
        return candidates

    return sorted(
        candidates,
        key=lambda candidate: (
            keyword_overlap_count(query_tokens, candidate),
            float(candidate.get("score") or 0),
        ),
        reverse=True,
    )


def keyword_overlap_count(query_tokens: set[str], candidate: dict[str, Any]) -> int:
    evidence_tokens = set(helpers.tokenize(f"{candidate.get('title', '')} {candidate.get('text', '')}"))
    return len(query_tokens & evidence_tokens)


def relevant_matches(query: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    min_score = relevance_threshold()
    score_filtered = [match for match in matches if float(match.get("score") or 0) >= min_score]
    if not score_filtered:
        return []

    query_tokens = set(helpers.tokenize(query))
    if not query_tokens:
        return score_filtered

    evidence_tokens = set()
    for match in score_filtered:
        evidence_tokens.update(helpers.tokenize(f"{match.get('title', '')} {match.get('text', '')}"))

    if query_tokens & evidence_tokens:
        return score_filtered

    logger.info("retrieval.rejected_no_keyword_overlap query_tokens=%s", sorted(query_tokens))
    return []


def relevance_threshold() -> float:
    try:
        return float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.18"))
    except ValueError:
        logger.warning("invalid RAG_MIN_RELEVANCE_SCORE; using default 0.18")
        return 0.18


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
    if not matches:
        return helpers.unknown_answer(), "no-relevant-evidence"

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


app = create_app(run_pipeline=run_pipeline, document_from_upload=helpers.document_from_upload, preload_demo_document=preload_demo_document)

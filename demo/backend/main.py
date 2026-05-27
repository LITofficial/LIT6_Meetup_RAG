import asyncio
import hashlib
import math
import os
import re
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pydantic import BaseModel, Field


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


# 1단계: PDF 업로드/텍스트 추출
# 발표에서는 "PDF를 모델에 바로 넣는 것이 아니라 먼저 읽을 수 있는 텍스트로 바꾼다"는 부분입니다.
async def document_from_upload(file: UploadFile) -> dict[str, str]:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    filename = file.filename or "uploaded.pdf"
    content_type = file.content_type or ""

    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(BytesIO(contents))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = contents.decode("utf-8", errors="ignore")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="파일에서 읽을 수 있는 텍스트를 찾지 못했습니다.")

    digest = hashlib.sha1(contents).hexdigest()[:10]
    return {
        "id": f"upload_{digest}",
        "filename": filename,
        "title": filename.rsplit(".", 1)[0],
        "text": text,
    }


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
    if not api_key.isascii() or not api_key.startswith("AIza"):
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY가 올바른 Gemini API key 형식이 아닙니다. .env에 AIza로 시작하는 실제 키를 넣어주세요.",
        )
    return api_key


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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=360,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", "다. ", " ", ""],
    )
    split_docs = splitter.split_documents(documents)
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
            }
        )

    return chunks


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


# 3단계: 임베딩 모델 준비
# LangChain GoogleGenerativeAIEmbeddings로 청크와 질문을 같은 벡터 공간에 올립니다.
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=require_google_api_key(),
    )


# 3단계: 임베딩/벡터 저장
# 각 청크를 Gemini 임베딩으로 바꾸고, 검색에 필요한 metadata와 함께 메모리 벡터 저장소를 만듭니다.
async def build_vector_store(
    chunks: list[dict[str, Any]],
    use_gemini: bool,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    if not use_gemini:
        return [{**chunk, "embedding": local_demo_embedding(chunk["text"])} for chunk in chunks]

    embeddings = get_embedding_model()
    try:
        vectors = await embeddings.aembed_documents([chunk["text"] for chunk in chunks])
    except GoogleGenerativeAIError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini 임베딩 생성에 실패했습니다. GEMINI_EMBEDDING_MODEL 값을 확인하세요. details={error}",
        ) from error
    return [
        {**chunk, "embedding": [round(value, 4) for value in vector[:12]], "full_embedding": vector}
        for chunk, vector in zip(chunks, vectors)
    ]


# 4단계: 검색
# 사용자 질문도 임베딩한 뒤, 벡터 저장소에서 의미적으로 가까운 Top-K 청크를 고릅니다.
async def retrieve(
    query: str,
    top_k: int,
    vector_store: list[dict[str, Any]],
    use_gemini: bool,
) -> list[dict[str, Any]]:
    if use_gemini:
        embeddings = get_embedding_model()
        query_embedding = await embeddings.aembed_query(query)
    else:
        query_embedding = local_demo_embedding(query)

    scored = []

    for item in vector_store:
        item_embedding = item.get("full_embedding") or item["embedding"]
        score = cosine_similarity(query_embedding, item_embedding)
        scored.append({**item, "score": round(score, 4)})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


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

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
    )
    chain = RAG_PROMPT | llm
    response = await chain.ainvoke({"query": query, "context": build_context(matches)})
    return clean_answer(str(response.content)), model_name


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


# 전체 RAG 파이프라인
# PDF 텍스트 -> LangChain Document -> 청킹 -> 임베딩 -> 검색 -> 프롬프트 조립 -> Gemini 답변까지 순서대로 실행합니다.
async def run_pipeline(
    query: str,
    top_k: int = 3,
    documents: list[dict[str, str]] | None = None,
    use_gemini: bool = True,
) -> QueryResponse:
    documents = documents or SAMPLE_DOCUMENTS
    langchain_documents = build_langchain_documents(documents)
    chunks = split_documents_with_langchain(langchain_documents)
    vector_store = await build_vector_store(chunks, use_gemini)
    matches = await retrieve(query, top_k, vector_store, use_gemini)
    if use_gemini:
        answer, model_name = await generate_answer_with_gemini(query, matches)
    else:
        answer = compose_fallback_answer(query, matches)
        model_name = "local-demo"

    prompt_preview = {
        "system": "LangChain ChatPromptTemplate: 검색된 근거만 사용해 답변하고 출처를 포함합니다.",
        "query": query,
        "contexts": [match["text"] for match in matches],
        "prompt_preview": build_prompt(query, matches),
    }

    stages = [
        PipelineStage(
            id="ingest",
            title="문서 수집",
            input=[document["filename"] for document in documents],
            output=[{"id": document["id"], "title": document["title"]} for document in documents],
            metric=f"{len(documents)} files",
        ),
        PipelineStage(
            id="chunk",
            title="청킹",
            input=[document["title"] for document in documents],
            output=[{"id": chunk["id"], "text": chunk["text"]} for chunk in chunks],
            metric=f"{len(chunks)} chunks by LangChain",
        ),
        PipelineStage(
            id="embed",
            title="임베딩",
            input=[chunk["id"] for chunk in chunks],
            output=[
                {"id": item["id"], "embedding_preview": item["embedding"][:4]}
                for item in vector_store
            ],
            metric=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001") if use_gemini else "local-demo",
        ),
        PipelineStage(
            id="store",
            title="벡터 저장",
            input=[{"chunk_id": item["id"], "metadata": item["filename"]} for item in vector_store],
            output={"collection": "demo_documents", "index": "cosine"},
            metric=f"{len(vector_store)} vectors",
        ),
        PipelineStage(
            id="retrieve",
            title="검색",
            input={"query": query, "top_k": top_k},
            output=[
                {
                    "chunk_id": match["id"],
                    "score": match["score"],
                    "text": match["text"],
                }
                for match in matches
            ],
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

    return QueryResponse(answer=answer, stages=stages, sources=sources, model=model_name)


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


# 실제 데모용 엔드포인트: PDF 파일과 질문을 받아 전체 LangChain RAG 파이프라인을 실행합니다.
@app.post("/api/rag/upload-query", response_model=QueryResponse)
async def query_uploaded_pdf(
    file: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(3),
) -> QueryResponse:
    document = await document_from_upload(file)
    return await run_pipeline(query=query, top_k=top_k, documents=[document])


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

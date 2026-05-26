import asyncio
import hashlib
import math
import os
import re
from collections import Counter
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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


class QueryRequest(BaseModel):
    query: str = Field(default="신규 고객 환불 정책을 근거와 함께 알려줘", min_length=1)
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


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def chunk_document(document: dict[str, str], chunk_size: int = 72) -> list[dict[str, Any]]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s*", document["text"]) if sentence.strip()]
    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []

    for sentence in sentences:
        if sum(len(item) for item in buffer) + len(sentence) > chunk_size and buffer:
            chunks.append(build_chunk(document, len(chunks) + 1, " ".join(buffer)))
            buffer = []
        buffer.append(sentence)

    if buffer:
        chunks.append(build_chunk(document, len(chunks) + 1, " ".join(buffer)))

    return chunks


def build_chunk(document: dict[str, str], index: int, text: str) -> dict[str, Any]:
    return {
        "id": f"{document['id']}_chunk_{index:03d}",
        "document_id": document["id"],
        "filename": document["filename"],
        "title": document["title"],
        "text": text,
        "tokens": tokenize(text),
    }


def embed_text(text: str, dimensions: int = 12) -> list[float]:
    vector = [0.0] * dimensions
    counts = Counter(tokenize(text))

    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1 if digest[2] % 2 == 0 else -1
        vector[index] += sign * (1 + math.log(count))

    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / magnitude, 4) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def build_vector_store(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    return [{**chunk, "embedding": embed_text(chunk["text"])} for chunk in chunks]


def retrieve(query: str, top_k: int, vector_store: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_embedding = embed_text(query)
    scored = []

    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({**item, "score": round(score, 4)})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def build_prompt(query: str, matches: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"근거 {index + 1}\n"
        f"- 파일: {match['filename']}\n"
        f"- 제목: {match['title']}\n"
        f"- 내용: {match['text']}"
        for index, match in enumerate(matches)
    )

    return (
        "너는 업로드된 문서를 바탕으로 답변하는 한국어 RAG 어시스턴트다.\n"
        "아래 규칙을 반드시 지켜라.\n"
        "1. 제공된 근거 내용만 사용한다.\n"
        "2. 답변은 자연스러운 한국어 문단으로 작성한다.\n"
        "3. LaTeX 수식 표기, Markdown 굵게 표시, 글머리표, [1] 같은 각주 번호를 쓰지 않는다.\n"
        "4. 수식은 일반 텍스트로 풀어 쓴다. 예: omega_avg = delta theta / delta t\n"
        "5. 출처는 답변 마지막 줄에 '출처: 파일명' 형식으로 한 번만 쓴다.\n"
        "6. 근거가 부족하면 부족하다고 말하고 추측하지 않는다.\n\n"
        f"사용자 질문:\n{query}\n\n"
        f"검색된 근거:\n{context}\n\n"
        "최종 답변:"
    )


async def generate_answer_with_gemini(query: str, matches: list[dict[str, Any]]) -> tuple[str, str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY가 설정되지 않았습니다. backend 환경변수에 Gemini API key를 추가하세요.",
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as error:
        raise HTTPException(
            status_code=500,
            detail="langchain-google-genai 패키지가 설치되지 않았습니다. backend 이미지를 다시 빌드하세요.",
        ) from error

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
    )
    response = await llm.ainvoke(build_prompt(query, matches))
    return clean_answer(str(response.content)), model_name


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


async def run_pipeline(
    query: str,
    top_k: int = 3,
    documents: list[dict[str, str]] | None = None,
    use_gemini: bool = True,
) -> QueryResponse:
    documents = documents or SAMPLE_DOCUMENTS
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    vector_store = [{**chunk, "embedding": embed_text(chunk["text"])} for chunk in chunks]
    matches = retrieve(query, top_k, vector_store)
    if use_gemini:
        answer, model_name = await generate_answer_with_gemini(query, matches)
    else:
        answer = compose_fallback_answer(query, matches)
        model_name = "local-demo"

    prompt_preview = {
        "system": "검색된 근거만 사용해 답변하고 출처를 포함합니다.",
        "query": query,
        "contexts": [match["text"] for match in matches],
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
            metric=f"{len(chunks)} chunks",
        ),
        PipelineStage(
            id="embed",
            title="임베딩",
            input=[chunk["id"] for chunk in chunks],
            output=[
                {"id": item["id"], "embedding_preview": item["embedding"][:4]}
                for item in vector_store
            ],
            metric="12 dims demo",
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/rag/demo", response_model=QueryResponse)
async def get_demo_pipeline() -> QueryResponse:
    return await run_pipeline(query="신규 고객 환불 정책을 근거와 함께 알려줘")


@app.post("/api/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    return await run_pipeline(query=request.query, top_k=request.top_k)


@app.post("/api/rag/upload-query", response_model=QueryResponse)
async def query_uploaded_pdf(
    file: UploadFile = File(...),
    query: str = Form("문서 내용을 요약하고 핵심 근거를 알려줘"),
    top_k: int = Form(3),
) -> QueryResponse:
    document = await document_from_upload(file)
    return await run_pipeline(query=query, top_k=top_k, documents=[document])


@app.websocket("/ws/rag")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query") or "신규 고객 환불 정책을 근거와 함께 알려줘"
            top_k = int(data.get("top_k") or 3)
            await stream_pipeline(websocket, query=query, top_k=top_k)
    except WebSocketDisconnect:
        return

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from schemas import QueryRequest, QueryResponse


PipelineRunner = Callable[..., Awaitable[QueryResponse]]
UploadReader = Callable[[UploadFile], Awaitable[dict[str, Any]]]


def create_app(run_pipeline: PipelineRunner, document_from_upload: UploadReader) -> FastAPI:
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
        files: list[UploadFile] = File(...),
        query: str = Form(...),
        top_k: int = Form(3),
    ) -> QueryResponse:
        if not files:
            raise HTTPException(status_code=400, detail="업로드할 파일이 필요합니다.")

        documents = [await document_from_upload(upload_file) for upload_file in files]
        return await run_pipeline(query=query, top_k=top_k, documents=documents)

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

    return app

from typing import Any

from pydantic import BaseModel, Field


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

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate


logger = logging.getLogger("uvicorn.error")

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
            "6. 검색된 근거 안에서 직접 확인할 수 없는 내용은 절대 추측하지 않는다.\n"
            "7. 근거가 없거나 질문에 직접 답할 문장이 부족하면 반드시 "
            "'문서에서 관련 근거를 찾지 못했습니다.'라고 답한다.",
        ),
        (
            "human",
            "사용자 질문:\n{query}\n\n검색된 근거:\n{context}\n\n최종 답변:",
        ),
    ]
)

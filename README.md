# Microsoft Foundry 기반 사용자 데이터 활용 RAG 시스템

## 프로젝트 개요

본 프로젝트는 Microsoft Learn의 **Azure AI Foundry**, **Copilot Studio**, **Azure AI Search**, **Azure OpenAI** 학습 내용을 기반으로 사용자 고유 데이터를 활용한 RAG(Retrieval-Augmented Generation) 기반 AI 응답 시스템을 구현하는 것을 목표로 한다.

이 시스템은 사용자가 보유한 문서를 AI와 연결하고, 검색 기반 Grounding을 수행한 뒤, 문서에 근거한 응답을 생성하는 간단한 Copilot 형태의 AI 시스템이다.

## 프로젝트 목표

### 목적

- 생성형 AI의 한계 보완
- 사용자 문서 기반 응답 생성
- Grounded AI 구현
- Microsoft Foundry 및 Copilot Studio 활용 경험 확보

### 시스템 범위

#### 포함 기능

- 문서 업로드
- 사용자 데이터 연결
- 문서 검색 기반 응답
- Copilot 질문/응답
- Grounded Response 생성

#### 제외 기능

- Fine-tuning
- Multi-Agent
- 복잡한 Vector DB 설계
- 실시간 데이터 파이프라인
- 고급 자동화
- 대규모 엔터프라이즈 아키텍처

본 프로젝트는 Microsoft Learn 모듈 수준의 기능 구현을 목표로 하며, 복잡한 엔터프라이즈 시스템 구축보다는 사용자 데이터 기반 RAG 흐름을 이해하고 시연하는 데 중점을 둔다.

## 핵심 개념

### 1. 생성형 AI의 한계

기존 LLM은 다음과 같은 한계를 가진다.

- 최신 데이터 부족
- 내부 문서 접근 불가
- Hallucination 발생 가능

예를 들어 사내 문서, 내부 매뉴얼, 업무 절차서와 관련된 질문을 할 경우, 일반적인 LLM은 해당 문서에 접근할 수 없기 때문에 정확하지 않은 응답을 생성할 수 있다.

### 2. RAG(Retrieval-Augmented Generation)

RAG는 사용자의 질문과 관련된 문서를 먼저 검색한 뒤, 검색된 결과를 기반으로 AI가 응답을 생성하는 구조이다.

```text
사용자 질문
    ↓
관련 문서 검색
    ↓
검색 결과를 Prompt에 포함
    ↓
LLM 응답 생성
```

이를 통해 AI는 단순히 사전 학습된 지식에 의존하지 않고, 사용자가 제공한 문서와 데이터를 근거로 답변할 수 있다.

## 사용 기술

| 구성 요소 | 기술 |
| --- | --- |
| AI 플랫폼 | Azure AI Foundry |
| Copilot 구성 | Copilot Studio |
| 검색 기능 | Azure AI Search |
| LLM | Azure OpenAI |
| 문서 저장 | Blob Storage |

## 시스템 아키텍처

### 전체 흐름

```text
[사용자 문서]
(PDF / DOCX / 내부 매뉴얼)
        ↓
[Blob Storage 저장]
        ↓
[Azure AI Search 인덱싱]
        ↓
[검색 가능한 데이터 생성]
        ↓
[Copilot Studio 연결]
        ↓
[사용자 질문]
        ↓
[관련 문서 검색]
        ↓
[Grounded Prompt 생성]
        ↓
[LLM 응답 생성]
        ↓
[사용자 응답 반환]
```

### 데이터 처리 흐름

#### 1. 문서 업로드

사용자는 PDF, DOCX, 내부 매뉴얼 등 AI가 참고할 문서를 업로드한다.

#### 2. 문서 저장

업로드된 문서는 Blob Storage에 저장된다. Blob Storage는 문서 원본을 보관하고, 이후 Azure AI Search가 해당 문서를 인덱싱할 수 있도록 데이터 소스 역할을 한다.

#### 3. 문서 인덱싱

Azure AI Search는 문서를 분석하여 검색 가능한 형태로 변환한다.

본 프로젝트에서는 Chunking, Embedding, Vector Search 등의 내부 동작을 직접 복잡하게 구현하지 않고 Microsoft 플랫폼에서 제공하는 기본 기능을 활용한다.

#### 4. Copilot 연결

Copilot Studio에서 검색 인덱스를 연결하고, 사용자 질문이 입력되었을 때 문서 검색 결과를 활용하도록 질문 처리 흐름을 구성한다.

## 질문 처리 흐름

```text
사용자 질문 입력
        ↓
Azure AI Search 수행
        ↓
관련 문서 검색
        ↓
검색 결과를 Prompt에 추가
        ↓
Azure OpenAI 응답 생성
        ↓
사용자에게 응답 반환
```

## Grounding 개념

Grounding은 AI가 사전에 학습된 일반 지식만으로 응답하지 않고, 검색된 사용자 데이터를 기반으로 응답하도록 만드는 과정이다.

Grounding을 적용하면 다음과 같은 효과를 기대할 수 있다.

- 정확도 향상
- Hallucination 감소
- 신뢰 가능한 응답 생성
- 내부 문서 기반 답변 제공

## 시연 목표

본 프로젝트 시연에서는 다음 흐름을 확인한다.

### 1. 문서 업로드

사용자 문서를 시스템에 등록한다.

### 2. Copilot 연결

Copilot Studio에서 문서 검색 기능을 활성화하고 Azure AI Search 인덱스와 연결한다.

### 3. 질문 입력

예시 질문:

```text
댐 유지보수 절차를 알려줘
```

### 4. Grounded Response 확인

AI가 내부 문서를 기반으로 관련 내용을 검색하고, 검색된 문서 내용을 근거로 응답을 생성하는 과정을 확인한다.

## 기대 효과

### 업무 효율 향상

- 문서 검색 시간 감소
- 반복 문의 감소
- 업무 지식 접근성 향상

### 지식 관리 강화

- 내부 데이터 활용 가능
- 문서 기반 AI 응답 제공
- 조직 내 지식 자산 활용도 증가

### AI 활용 확대

- 내부 Assistant 구축
- 기술 문서 검색
- 교육 지원 시스템 구현

## 한계점

### 현재 한계

- 데이터 품질 의존성
- 검색 정확도 제한
- 복잡한 추론 기능 부족
- Microsoft 플랫폼 기본 기능에 대한 의존성

### 향후 개선 방향

- 멀티모달 데이터 지원
- 실시간 데이터 연동
- 고급 Agent 기능 추가
- Prompt 최적화
- 검색 품질 개선

## 발표 핵심 메시지

생성형 AI는 단순한 채팅 시스템이 아니라, 사용자 데이터를 연결했을 때 실제 업무에 활용 가능한 AI 시스템이 된다.

본 프로젝트는 Microsoft Foundry와 Copilot Studio를 활용하여 사용자 문서 기반 RAG 구조를 구성하고, 검색된 문서에 근거한 신뢰 가능한 AI 응답을 생성하는 과정을 보여준다.

## 참고 자료

- Microsoft Learn
- Azure AI Foundry
- Copilot Studio
- Azure AI Search
- Azure OpenAI

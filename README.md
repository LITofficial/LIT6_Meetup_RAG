<!--
---
name: User Data Grounded RAG Copilot
description: A presentation README about building a document-grounded copilot with Microsoft Copilot Studio, Azure AI Search, Azure OpenAI, and Microsoft AI Foundry.
products:
- microsoft-copilot-studio
- azure-ai-search
- azure-openai
- azure-ai-foundry
page_type: sample
---
-->

<p align="center">
  <h1 align="center">LIT 6월 밋업 - 사용자 데이터 기반 Grounded RAG Copilot</h1>
</p>

<p align="center">
  Microsoft Copilot Studio와 Azure AI Search로<br/>
  내 문서를 근거로 답변하는 AI 에이전트 만들기
</p>

---

## 발표 주제

이번 발표에서는 Microsoft Learn의 [Build a copilot with Azure AI Studio - 소개](https://learn.microsoft.com/ko-kr/training/modules/build-copilot-ai-studio/1-introduction)를 바탕으로, 사용자가 가진 문서를 AI와 연결해 **근거 있는 답변을 생성하는 Copilot**을 만드는 과정을 다룹니다.

핵심은 단순합니다.

> AI에게 더 많은 말을 시키는 것이 아니라, AI가 답변할 때 참고할 수 있는 "근거 데이터"를 연결하는 것입니다.

일반적인 LLM은 자연스러운 문장을 만들 수 있지만, 우리 조직의 내부 문서, 최신 정책, 매뉴얼, 회의록, 업무 절차서까지 자동으로 알고 있지는 않습니다. 그래서 실제 업무에서는 그럴듯하지만 틀린 답변이 나오거나, 최신 문서와 맞지 않는 답변이 나올 수 있습니다.

이번 발표는 이 문제를 **RAG(Retrieval-Augmented Generation)** 와 **Grounding** 개념으로 해결하는 흐름을 설명합니다.

---

# 문제상황 제시

## AI가 답은 잘하지만, 근거를 모르는 상황

여러분은 이런 경험이 있으실 겁니다.

회사 규정, 수업 자료, 회의록, 과제 문서, 제품 매뉴얼처럼 분명히 어딘가에 있는 자료를 찾고 싶은데 파일명이 기억나지 않습니다. 내용은 대충 기억나지만, 정확한 문서 위치를 몰라서 폴더를 하나씩 열어보거나 검색창에 여러 단어를 넣어가며 찾게 됩니다.

이런 상황에서 일반 챗봇에게 질문하면 자연스럽게 답은 해줄 수 있습니다. 하지만 그 답변이 실제 문서에 근거한 것인지는 알기 어렵습니다.

| 사용자가 원하는 것 | 일반 LLM만으로 어려운 점 |
| --- | --- |
| 내부 문서 기반 답변 | 모델이 해당 문서를 모를 수 있음 |
| 최신 정책 반영 | 학습 시점 이후 정보는 반영되지 않을 수 있음 |
| 출처가 있는 답변 | 답변의 근거 문서를 확인하기 어려움 |
| 조직별 업무 절차 안내 | 공개 데이터에 없는 내부 지식은 부족함 |

즉, 문제의 핵심은 AI가 말을 못 하는 것이 아닙니다.

> **AI가 우리 데이터를 근거로 답하고 있는지 확인하기 어렵다는 것입니다.**

---

# 해결 방향

## 사용자 데이터를 AI의 근거로 연결하기

이 문제를 해결하려면 AI가 답변하기 전에 먼저 관련 문서를 찾아야 합니다. 그리고 찾은 문서를 바탕으로 답변해야 합니다.

이 구조를 RAG라고 부릅니다.

```text
사용자 질문
  -> 관련 문서 검색
  -> 검색 결과를 프롬프트에 포함
  -> 언어 모델이 근거 기반 답변 생성
  -> 사용자에게 답변 반환
```

이렇게 하면 AI는 기억에만 의존하지 않고, 사용자가 제공한 문서와 검색 결과를 바탕으로 답변할 수 있습니다.

이번 발표에서는 이 흐름을 Microsoft Copilot Studio, Azure AI Search, Azure OpenAI, Microsoft AI Foundry를 중심으로 설명합니다.

---

# 핵심 개념

## 1. Grounding

Grounding은 AI가 일반적인 학습 지식만으로 답하지 않고, 연결된 데이터 원본에서 찾은 실제 정보를 기반으로 답변하도록 만드는 방식입니다.

예를 들어 "휴가 신청 절차를 알려줘"라는 질문이 들어왔을 때, 모델이 임의로 일반적인 휴가 절차를 만들어내는 것이 아니라 사내 인사 규정 문서에서 관련 내용을 찾고 그 내용을 바탕으로 답변하는 것입니다.

Grounding을 적용하면 다음 효과를 기대할 수 있습니다.

- 근거 없는 추측 감소
- 내부 문서 기반 답변 생성
- 최신 문서 반영 가능
- 답변 신뢰도 향상

## 2. RAG

RAG는 Retrieval-Augmented Generation의 약자입니다. 한국어로는 보통 검색 증강 생성이라고 설명할 수 있습니다.

의미는 간단합니다.

```text
Retrieval
  -> 질문과 관련된 정보를 먼저 검색

Augmented
  -> 검색한 정보를 모델 입력에 보강

Generation
  -> 보강된 정보를 바탕으로 답변 생성
```

RAG는 모델을 다시 학습시키는 방식이 아닙니다. 문서를 검색해 필요한 정보를 가져오고, 그 정보를 모델이 답변에 활용하도록 만드는 방식입니다.

## 3. Copilot Studio

Copilot Studio는 사용자가 직접 에이전트를 만들고, 지식 원본을 연결하고, 실제 채널에 배포할 수 있게 해주는 Microsoft의 low-code AI 에이전트 제작 도구입니다.

이번 발표에서는 Copilot Studio를 사용해 "사용자 질문을 받고, 연결된 문서를 참고해 답변하는 Copilot"의 구조를 설명합니다.

## 4. Azure AI Search

Azure AI Search는 문서를 검색 가능한 형태로 만들고, 사용자의 질문과 관련된 정보를 찾아주는 역할을 합니다.

단순히 파일명만 검색하는 것이 아니라, 문서 안의 텍스트와 의미를 기반으로 관련 결과를 찾을 수 있습니다. 필요에 따라 OCR, 인덱서, skillset 같은 기능을 활용해 PDF, 이미지, 문서 자료를 검색 가능한 지식 자산으로 바꿀 수 있습니다.

---

# 전체 구조

## 시스템 흐름

```text
[사용자 문서]
  -> [Azure Blob Storage 저장]
  -> [Azure AI Search 인덱싱]
  -> [Copilot Studio 지식 원본 연결]
  -> [사용자 질문 입력]
  -> [관련 문서 검색]
  -> [검색 결과를 근거로 프롬프트 구성]
  -> [Azure OpenAI 답변 생성]
  -> [사용자에게 답변 제공]
```

이 구조에서 중요한 점은 모델이 모든 문서를 외우고 있다고 가정하지 않는다는 것입니다.

문서는 별도의 지식 원본으로 관리하고, 질문이 들어올 때마다 관련 내용을 검색해 답변에 활용합니다.

---

# 기술 구성

| 구성 요소 | 역할 |
| --- | --- |
| Microsoft Copilot Studio | 사용자가 대화하는 Copilot 구성 |
| Azure AI Search | 문서 인덱싱 및 관련 정보 검색 |
| Azure OpenAI | 검색된 근거를 바탕으로 자연어 답변 생성 |
| Azure Blob Storage | 문서 원본 저장 |
| Microsoft AI Foundry | AI 서비스 개발과 확장을 위한 플랫폼 |

---

# 데모 시나리오

## 데모 목표

이번 데모의 목표는 단순히 AI가 답변하는 모습을 보여주는 것이 아닙니다.

> 사용자가 올린 문서를 AI가 검색하고, 그 검색 결과를 근거로 답변하는 흐름을 확인하는 것입니다.

## 사용자 관점 흐름

```text
1. 사용자가 업무 문서나 매뉴얼을 준비한다.
2. 문서를 저장소에 업로드한다.
3. Azure AI Search가 문서를 검색 가능한 형태로 인덱싱한다.
4. Copilot Studio에서 지식 원본을 연결한다.
5. 사용자가 자연어로 질문한다.
6. Copilot이 관련 문서를 찾는다.
7. 검색된 내용을 바탕으로 답변한다.
```

## 예시 질문

```text
휴가 신청 절차를 알려줘.
```

```text
장비 반납 규정은 어떻게 되나요?
```

```text
신입 구성원이 먼저 확인해야 할 문서는 무엇인가요?
```

```text
회의록에서 지난주에 결정된 작업 항목을 정리해줘.
```

---

# 일반 검색과 RAG Copilot의 차이

| 구분 | 일반 검색 | RAG 기반 Copilot |
| --- | --- | --- |
| 입력 방식 | 키워드 중심 | 자연어 질문 |
| 결과 형태 | 문서 목록 | 문서 기반 답변 |
| 문맥 이해 | 제한적 | 질문 의도 기반 검색 가능 |
| 답변 생성 | 사용자가 직접 문서를 읽어야 함 | AI가 문서를 바탕으로 요약 및 설명 |
| 활용 목적 | 파일 찾기 | 업무 질문 답변 |

일반 검색은 "어떤 문서를 봐야 하는지"를 알려주는 데 가깝습니다. RAG Copilot은 "그 문서에 따르면 답이 무엇인지"까지 도와주는 방향에 가깝습니다.

---

# 발표 구성

1. 문제상황 제시: AI 답변은 자연스럽지만 근거가 부족한 문제
2. 일반 LLM의 한계: 최신성, 내부 문서 접근, hallucination
3. Grounding과 RAG 개념 설명
4. Azure AI Search와 Copilot Studio의 역할
5. 전체 아키텍처 흐름
6. 데모 시나리오
7. 한계와 개선 방향
8. 마무리 메시지

---

# 한계와 개선 방향

이번 발표는 Microsoft Learn 모듈 수준의 개념과 데모 흐름을 이해하는 데 초점을 둡니다. 실제 서비스로 운영하려면 다음 요소를 추가로 고려해야 합니다.

- 문서 권한과 사용자별 접근 제어
- 답변에 사용된 출처 표시
- 검색 품질 평가와 인덱스 튜닝
- 데이터 최신성 관리
- 민감 정보 필터링
- 프롬프트 최적화
- 운영 로그와 사용자 피드백 수집

RAG를 적용한다고 해서 모든 문제가 자동으로 해결되는 것은 아닙니다. 중요한 것은 어떤 데이터를 연결할지, 검색 품질을 어떻게 관리할지, 답변 근거를 어떻게 검증할지입니다.

---

# 결론

생성형 AI를 실제 업무에 활용하려면 모델의 성능만큼이나 중요한 것이 있습니다.

> **AI가 어떤 데이터를 근거로 판단하고 답변하는가입니다.**

이번 발표에서 다루는 Grounded RAG Copilot은 사용자가 가진 문서와 AI를 연결해, 단순한 챗봇을 업무 지식 기반 에이전트로 확장하는 방법을 보여줍니다.

결국 핵심은 AI에게 모든 것을 외우게 만드는 것이 아닙니다.

**현실의 문서와 데이터를 검색 가능한 지식으로 만들고, 그 지식을 AI의 답변 과정에 연결하는 것입니다.**

---

## 참고 자료

- [Microsoft Learn - Build a copilot with Azure AI Studio: 소개](https://learn.microsoft.com/ko-kr/training/modules/build-copilot-ai-studio/1-introduction)
- [Microsoft Copilot Studio 설명서](https://learn.microsoft.com/ko-kr/microsoft-copilot-studio/)
- [Azure AI Search 설명서](https://learn.microsoft.com/ko-kr/azure/search/)
- [Azure OpenAI Service 설명서](https://learn.microsoft.com/ko-kr/azure/ai-services/openai/)


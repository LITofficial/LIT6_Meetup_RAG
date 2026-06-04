import { useState } from 'react'
import StageFocus from './components/StageFocus'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'
const REQUEST_TIMEOUT_MS = 90000

const defaultStages = [
  {
    id: 'chunk',
    title: '청킹',
    label: 'Chunking',
    summary: '긴 문서를 검색 가능한 작은 단락으로 나눕니다.',
    input: ['원문 텍스트', '문서 제목', '페이지 번호'],
    output: ['chunk_001: 환불 규정', 'chunk_002: 배송 예외', 'chunk_003: 보안 정책'],
    payload: '청크',
    metric: '420 tokens',
  },
  {
    id: 'embed',
    title: '임베딩',
    label: 'Embedding',
    summary: '각 chunk를 AI가 비교할 수 있는 숫자표로 바꿉니다.',
    input: ['chunk_001', 'chunk_002', 'chunk_003'],
    output: ['[0.12, -0.31, 0.77, ...]', '[0.08, -0.24, 0.69, ...]'],
    payload: '벡터화',
    metric: 'vectorized',
  },
  {
    id: 'store',
    title: '벡터 저장',
    label: 'Vector DB',
    summary: '벡터와 원문 위치를 함께 저장합니다.',
    input: ['embedding', 'chunk text', 'metadata'],
    output: ['collection: docs', 'index: cosine', 'metadata filter: team=ops'],
    payload: '저장',
    metric: 'cosine',
  },
  {
    id: 'retrieve',
    title: '검색',
    label: 'Retriever',
    summary: '질문과 가장 비슷한 chunk를 점수순으로 찾습니다.',
    input: ['질문: 환불은 언제 가능해?', 'query embedding'],
    output: ['score 0.91: 환불 규정', 'score 0.84: 배송 예외', 'score 0.78: 고객 지원'],
    payload: 'Top-K',
    metric: '0.91',
  },
  {
    id: 'augment',
    title: '컨텍스트 조립',
    label: 'Prompting',
    summary: '검색 결과를 질문과 함께 모델 입력 프롬프트로 묶습니다.',
    input: ['사용자 질문', '검색된 청크 3개', '시스템 지시문'],
    output: ['근거 중심 프롬프트', '출처 목록', '답변 제약 조건'],
    payload: '프롬프트',
    metric: '2.1k tokens',
  },
  {
    id: 'generate',
    title: '답변 생성',
    label: 'LLM',
    summary: '모델이 조립된 컨텍스트만 근거로 답변을 생성합니다.',
    input: ['final prompt', 'retrieved context'],
    output: ['답변 초안', '인용 출처', '신뢰도 표시'],
    payload: '답변',
    metric: 'grounded',
  },
]

const stageCopy = {
  chunk: {
    label: 'Chunking',
    summary: '추출된 텍스트를 검색 가능한 작은 청크로 나눕니다.',
    payload: '청크',
  },
  embed: {
    label: 'Embedding',
    summary: '각 chunk를 AI가 비교할 수 있는 숫자표로 바꿉니다.',
    payload: '벡터화',
  },
  store: {
    label: 'Vector DB',
    summary: '청크, 벡터, 파일명을 함께 저장해 검색 가능한 컬렉션을 만듭니다.',
    payload: '저장',
  },
  retrieve: {
    label: 'Retriever',
    summary: '질문과 가장 비슷한 chunk를 점수순으로 찾습니다.',
    payload: 'Top-K',
  },
  augment: {
    label: 'Prompting',
    summary: '검색된 근거 청크를 사용자 질문과 함께 최종 프롬프트로 조립합니다.',
    payload: '프롬프트',
  },
  generate: {
    label: 'LLM',
    summary: '조립된 프롬프트와 근거만 사용해 답변을 생성합니다.',
    payload: '답변',
  },
}

const connections = [
  '임베딩',
  '저장',
  '검색',
  '조립',
  '생성',
]

function App() {
  const [stages, setStages] = useState(defaultStages)
  const [activeStage, setActiveStage] = useState(0)
  const [query, setQuery] = useState('')
  const [selectedFiles, setSelectedFiles] = useState([])
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [modelName, setModelName] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [showDebug, setShowDebug] = useState(false)
  const [hasRun, setHasRun] = useState(false)
  const [isInputOpen, setIsInputOpen] = useState(true)
  const stage = stages[activeStage]
  const isGenerateStage = stage?.id === 'generate'
  const shouldShowInputForm = isInputOpen || !hasRun || isSubmitting
  const stepLabel = `${activeStage + 1} / ${stages.length}`

  function moveStage(direction) {
    setActiveStage((current) => {
      const next = current + direction
      if (next < 0) return stages.length - 1
      if (next >= stages.length) return 0
      return next
    })
  }

  function normalizeStages(apiStages) {
    return apiStages.map((item) => {
      const copy = stageCopy[item.id] ?? {}
      return {
        id: item.id,
        title: item.title,
        label: copy.label ?? item.id,
        summary: copy.summary ?? item.title,
        rawInput: item.input,
        rawOutput: item.output,
        input: toDisplayItems(item.input),
        output: toDisplayItems(item.output),
        payload: copy.payload ?? item.metric,
        metric: item.metric,
      }
    })
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setIsSubmitting(true)

    try {
      const response = selectedFiles.length > 0
        ? await submitUploadedPdf(selectedFiles, query)
        : await submitPrompt(query)

      if (!response.ok) {
        throw new Error(await getErrorMessage(response))
      }

      const data = await response.json()
      setStages(normalizeStages(data.stages))
      setAnswer(data.answer)
      setSources(data.sources ?? [])
      setModelName(data.model ?? '')
      setActiveStage(0)
      setHasRun(true)
      setIsInputOpen(false)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="rag-app">
      <header className="topbar">
        <div>
          <p className="eyebrow">RAG Pipeline Visualizer</p>
          <h1>데이터가 답변이 되는 과정</h1>
        </div>
        <div className="run-panel" aria-label="pipeline progress">
          <div className="run-actions">
            <span>{stage.title}</span>
            <strong>{stepLabel}</strong>
          </div>
          <div className="progress-track">
            <div style={{ width: `${((activeStage + 1) / stages.length) * 100}%` }} />
          </div>
        </div>
      </header>

      <section className="workspace">
        <aside className="stage-list" aria-label="RAG stages">
          {stages.map((item, index) => (
            <button
              key={item.id}
              type="button"
              className={index === activeStage ? 'stage-tab active' : 'stage-tab'}
              onClick={() => setActiveStage(index)}
            >
              <span className="stage-index">{String(index + 1).padStart(2, '0')}</span>
              <span>
                <strong>{item.title}</strong>
                <small>{item.label}</small>
              </span>
            </button>
          ))}
        </aside>

        <section className="flow-board" aria-label="visual data flow">
          {shouldShowInputForm ? (
            <form className="rag-form" onSubmit={handleSubmit}>
              <label className="file-picker">
                <span>PDF</span>
                <input
                  type="file"
                  accept="application/pdf,.pdf,.txt,.md"
                  multiple
                  onChange={(event) => setSelectedFiles(Array.from(event.target.files ?? []))}
                />
                <strong>{formatSelectedFiles(selectedFiles)}</strong>
              </label>
              <label className="prompt-box">
                <span>Prompt</span>
                <textarea
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  rows="3"
                  placeholder="PDF에 대해 묻고 싶은 내용을 입력하세요"
                />
              </label>
              <button type="submit" disabled={isSubmitting || !query.trim()}>
                {isSubmitting ? 'Running' : 'Run RAG'}
              </button>
            </form>
          ) : (
            <section className="run-summary">
              <div>
                <span>Current Run</span>
                <strong>{formatSelectedFiles(selectedFiles, '샘플 문서')} · “{query}”</strong>
              </div>
              <button type="button" onClick={() => setIsInputOpen(true)}>
                Edit input
              </button>
            </section>
          )}

          {error && <p className="error-message">{error}</p>}

          <div className="pipeline-map">
            {stages.map((item, index) => (
              <div key={item.id} className="map-segment">
                <button
                  type="button"
                  className={index === activeStage ? 'node active' : 'node'}
                  onClick={() => setActiveStage(index)}
                  aria-label={item.title}
                >
                  <span>{item.title}</span>
                  <strong>{item.label}</strong>
                </button>
                {index < stages.length - 1 && (
                  <div className={index < activeStage ? 'connector filled' : 'connector'}>
                    <span>{connections[index]}</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="packet-lane" aria-hidden="true">
            <div
              className="data-packet"
              style={{ left: `calc(${(activeStage / (stages.length - 1)) * 100}% - 42px)` }}
            >
              {stage.title}
            </div>
          </div>

          <div className="detail-grid">
            <StageFocus stage={stage} activeStage={activeStage} />
          </div>

          {isGenerateStage && (answer || sources.length > 0) && (
            <section className="result-panel">
              <div>
                <span>Final Answer {modelName ? `· ${modelName}` : ''}</span>
                <p>{answer}</p>
              </div>
              <div>
                <span>Sources</span>
                <ul>
                  {sources.map((source) => (
                    <li key={source.chunk_id}>
                      <strong>{source.filename}</strong>
                      <small>score {source.score}</small>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          )}

          <section className="debug-panel">
            <button type="button" onClick={() => setShowDebug((value) => !value)}>
              <span>{showDebug ? 'Hide' : 'Show'} Debug Details</span>
              <strong>Raw input / output</strong>
            </button>
            {showDebug && (
              <div className="debug-grid">
                <section className="detail-panel">
                  <div className="panel-heading">
                    <span>Raw Input</span>
                    <strong>{stage.title}에 들어간 값</strong>
                  </div>
                  <ul>
                    {stage.input.map((item, index) => (
                      <li key={`${stage.id}-input-${index}`}>{item}</li>
                    ))}
                  </ul>
                </section>

                <section className="detail-panel">
                  <div className="panel-heading">
                    <span>Raw Output</span>
                    <strong>전체 데이터 · 스크롤</strong>
                  </div>
                  <ul>
                    {stage.output.map((item, index) => (
                      <li key={`${stage.id}-output-${index}`}>{item}</li>
                    ))}
                  </ul>
                </section>
              </div>
            )}
          </section>

          <div className="controls">
            <button type="button" onClick={() => moveStage(-1)} aria-label="previous stage">
              ←
            </button>
            <div className="step-dots" aria-label="stage position">
              {stages.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  className={index === activeStage ? 'dot active' : 'dot'}
                  onClick={() => setActiveStage(index)}
                  aria-label={item.title}
                />
              ))}
            </div>
            <button type="button" onClick={() => moveStage(1)} aria-label="next stage">
              →
            </button>
          </div>
        </section>
      </section>
    </main>
  )
}

function toDisplayItems(value) {
  const items = Array.isArray(value) ? value : [value]

  return items.map((item) => {
    if (typeof item === 'string') return item
    if (typeof item === 'number') return String(item)
    if (item === null || item === undefined) return ''
    if (item.id && item.embedding_preview) {
      return `${item.id}\nembedding: [${item.embedding_preview.join(', ')}, ...]`
    }
    if (item.id && item.text) {
      return `${item.id}\n${item.text}`
    }
    if (item.chunk_id && item.score !== undefined) {
      return `${item.chunk_id}\nscore: ${item.score}\n${item.text ?? ''}`
    }
    return JSON.stringify(item, null, 2)
  })
}

function submitPrompt(query) {
  return fetchWithTimeout(`${API_BASE_URL}/api/rag/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: 3 }),
  })
}

function submitUploadedPdf(files, query) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  formData.append('query', query)
  formData.append('top_k', '3')

  return fetchWithTimeout(`${API_BASE_URL}/api/rag/upload-query`, {
    method: 'POST',
    body: formData,
  })
}

function formatSelectedFiles(files, emptyLabel = '파일 선택') {
  if (!files.length) return emptyLabel
  if (files.length === 1) return files[0].name
  return `${files[0].name} 외 ${files.length - 1}개`
}

async function fetchWithTimeout(url, options) {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('RAG 요청이 90초 안에 끝나지 않았습니다. PDF 추출, 임베딩, Gemini 응답 중 지연된 단계가 있는지 백엔드 로그를 확인하세요.')
    }
    throw error
  } finally {
    window.clearTimeout(timer)
  }
}

async function getErrorMessage(response) {
  try {
    const data = await response.json()
    if (typeof data.detail === 'string') return data.detail
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((item) => {
          if (typeof item === 'string') return item
          const location = Array.isArray(item.loc) ? item.loc.join('.') : ''
          return [location, item.msg].filter(Boolean).join(': ')
        })
        .join('\n')
    }
    if (data.detail) return JSON.stringify(data.detail)
    return 'RAG 요청에 실패했습니다.'
  } catch {
    return response.text()
  }
}

export default App

function StageFocus({ stage, activeStage }) {
  return (
    <section className="stage-focus">
      <div className="stage-focus-header">
        <span className="stage-number">{String(activeStage + 1).padStart(2, '0')}</span>
        <div>
          <h2>{stage.title}</h2>
          <p>{stage.summary}</p>
        </div>
      </div>

      <StageVisualization stage={stage} />

      <div className="metric-row">
        <span>{stage.metric}</span>
        <span>{stage.label}</span>
      </div>
    </section>
  )
}

function StageVisualization({ stage }) {
  if (stage.id === 'chunk') return <ChunkView stage={stage} />
  if (stage.id === 'embed') return <EmbeddingView vectors={asArray(stage.rawOutput ?? stage.output)} />
  if (stage.id === 'store') return <StoreView stage={stage} />
  if (stage.id === 'retrieve') return <RetrieveView stage={stage} />
  if (stage.id === 'augment') return <PromptView prompt={stage.rawOutput ?? stage.output} />
  if (stage.id === 'generate') return <GenerateView stage={stage} />
  return <FallbackView stage={stage} />
}

function ChunkView({ stage }) {
  const chunks = asArray(stage.rawOutput ?? stage.output)
  const sourceItems = asArray(stage.rawInput ?? stage.input)
  const source = sourceItems[0] ?? {}
  const sourceTitle = sourceItems.length > 1 ? `${sourceItems.length}개 문서` : source.filename ?? source.title ?? '원본 문서'
  const shownCount = Math.min(chunks.length, 6)
  const sourceText = String(source.text ?? sourceItems[0] ?? 'PDF에서 추출된 긴 텍스트')
  const sourceMeta = source.chars ? `전체 ${source.chars.toLocaleString()}자 중 앞부분 표시` : '원본 텍스트 일부'

  return (
    <div className="chunk-viz">
      <div className="chunk-transform" aria-label="long document becomes chunk list">
        <article className="document-preview">
          <div className="chunk-section-heading">
            <span>input</span>
            <strong>{sourceTitle}</strong>
          </div>
          <small>{sourceMeta}</small>
          <p>{shorten(sourceText, 220)}</p>
          <div className="cut-guide" aria-label="chunk boundaries">
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
        </article>

        <div className="chunk-flow" aria-label="document becomes chunks">
          <i />
          <span>문서가 청크 목록으로 변환됩니다</span>
        </div>

        <section className="chunk-list">
          <div className="chunk-section-heading">
            <span>output</span>
            <strong>{chunks.length || stage.metric} chunks · 대표 {shownCount}개</strong>
          </div>
          {chunks.slice(0, 6).map((chunk, index) => (
            <article className="chunk-card" key={chunk.id ?? index}>
              <span>chunk {String(index + 1).padStart(2, '0')}</span>
              <strong>{compactId(chunk.id ?? `chunk_${index + 1}`)}</strong>
              <p>{shorten(chunk.text ?? chunk, 70)}</p>
              <small>{estimateTokens(chunk.text ?? chunk)} tokens · metadata 포함</small>
            </article>
          ))}
          <p className="chunk-note">경계선 기준으로 자르고, 앞뒤 문맥 일부를 겹쳐 검색 근거가 끊기지 않게 보존합니다.</p>
        </section>
      </div>
    </div>
  )
}

function EmbeddingView({ vectors }) {
  const normalizedVectors = normalizeVectors(vectors)
  const plotItems = normalizedVectors
    .map((vector, index) => {
      const point = vector.point3d ?? vectorPoint3d(vector, index)
      const projected = projectPlotPoint(point)
      return {
        id: vector.id ?? `chunk_${index + 1}`,
        label: compactId(vector.id ?? `chunk_${index + 1}`),
        cluster: clusterIndex(point, index),
        index,
        point,
        projected,
      }
    })
    .sort((left, right) => left.point.z - right.point.z)
  const gridLines = buildPlotGrid()

  return (
    <div className="embedding-visual">
      <div className="embedding-space plot-space" aria-label="PCA 3D embedding plot">
        <svg className="embedding-plot" viewBox="0 0 1200 680" role="img" aria-label="PCA 3D embedding plot">
          <g className="plot-walls">
            {gridLines.map((line) => (
              <line
                key={line.id}
                className={line.kind}
                x1={line.start.x}
                y1={line.start.y}
                x2={line.end.x}
                y2={line.end.y}
              />
            ))}
          </g>
          <g className="plot-points">
            {plotItems.map((item) => (
              <g className={`plot-point cluster-${item.cluster}`} key={`plot-${item.id}`}>
                <circle
                  className="plot-point-halo"
                  cx={item.projected.x}
                  cy={item.projected.y}
                  r={10 + item.projected.depth * 6}
                />
                <circle
                  className="plot-point-core"
                  cx={item.projected.x}
                  cy={item.projected.y}
                  r={4.5 + item.projected.depth * 3.5}
                />
              </g>
            ))}
          </g>
        </svg>
      </div>
    </div>
  )
}

function buildPlotGrid() {
  const lines = []
  const values = [-36, -24, -12, 0, 12, 24, 36]

  values.forEach((value) => {
    lines.push(plotLine(`floor-x-${value}`, { x: -36, y: -36, z: value }, { x: 36, y: -36, z: value }, 'floor-line'))
    lines.push(plotLine(`floor-z-${value}`, { x: value, y: -36, z: -36 }, { x: value, y: -36, z: 36 }, 'floor-line'))
    lines.push(plotLine(`wall-y-left-${value}`, { x: -36, y: -36, z: value }, { x: -36, y: 36, z: value }, 'wall-line'))
    lines.push(plotLine(`wall-x-back-${value}`, { x: value, y: -36, z: 36 }, { x: value, y: 36, z: 36 }, 'wall-line'))
    lines.push(plotLine(`wall-y-back-${value}`, { x: value, y: -36, z: 36 }, { x: value, y: 36, z: 36 }, 'wall-line'))
  })

  return lines
}

function plotLine(id, startPoint, endPoint, kind) {
  return {
    id,
    kind,
    start: projectPlotPoint(startPoint),
    end: projectPlotPoint(endPoint),
  }
}

function RetrieveView({ stage }) {
  const retrieval = normalizeRetrieval(stage.rawOutput ?? stage.output)
  const candidates = retrieval.candidates
  const selectedMatches = retrieval.topK
  const query = typeof stage.rawInput === 'object' ? stage.rawInput.query : 'user query'
  const topK = typeof stage.rawInput === 'object' ? stage.rawInput.top_k : selectedMatches.length
  const topMatchIds = new Set(selectedMatches.map((match) => match.chunk_id))
  const queryPoint = stage.rawInput?.query_point3d ?? retrieval.queryPoint3d ?? { x: 0, y: 0, z: 0 }
  const queryProjection = projectPlotPoint(queryPoint)
  const plotItems = candidates
    .slice(0, 8)
    .map((match, index) => {
      const point = match.point3d ?? rankedPoint3d(index)
      return {
        ...match,
        point,
        projected: projectPlotPoint(point),
        cluster: clusterIndex(point, index),
        isTopMatch: topMatchIds.has(match.chunk_id),
      }
    })
    .sort((left, right) => left.point.z - right.point.z)
  const topLinks = selectedMatches
    .slice(0, topK)
    .map((match, index) => {
      const point = match.point3d ?? candidates.find((candidate) => candidate.chunk_id === match.chunk_id)?.point3d
      if (!point) return null
      return {
        id: match.chunk_id ?? `match_${index + 1}`,
        target: projectPlotPoint(point),
      }
    })
    .filter(Boolean)
  const gridLines = buildPlotGrid()
  const maxScore = Math.max(...candidates.map((match) => Number(match.score) || 0), 1)

  return (
    <div className="retrieve-viz">
      <div className="retrieve-map plot-space">
        <div className="map-explainer">
          <span>query embedding</span>
          <strong>질문 벡터와 가까운 chunk를 찾아 연결합니다</strong>
        </div>
        <svg className="retrieve-plot" viewBox="0 0 1200 680" role="img" aria-label="query vector and retrieved chunks">
          <g className="plot-walls">
            {gridLines.map((line) => (
              <line
                key={`retrieve-${line.id}`}
                className={line.kind}
                x1={line.start.x}
                y1={line.start.y}
                x2={line.end.x}
                y2={line.end.y}
              />
            ))}
          </g>
          <g className="query-links">
            {topLinks.map((link) => (
              <line
                key={`query-link-${link.id}`}
                className="query-link"
                x1={queryProjection.x}
                y1={queryProjection.y}
                x2={link.target.x}
                y2={link.target.y}
              />
            ))}
          </g>
          <g className="plot-points">
            {plotItems.map((item) => (
              <g
                className={`candidate-point cluster-${item.cluster}${item.isTopMatch ? ' top-match' : ''}`}
                key={`retrieve-point-${item.chunk_id ?? item.id}`}
              >
                <circle
                  className="plot-point-halo"
                  cx={item.projected.x}
                  cy={item.projected.y}
                  r={9 + item.projected.depth * 6}
                />
                <circle
                  className="plot-point-core"
                  cx={item.projected.x}
                  cy={item.projected.y}
                  r={4 + item.projected.depth * 3.5}
                />
              </g>
            ))}
            <g className="query-vector">
              <circle className="query-vector-halo" cx={queryProjection.x} cy={queryProjection.y} r="24" />
              <circle className="query-vector-core" cx={queryProjection.x} cy={queryProjection.y} r="12" />
              <text className="query-vector-label" x={queryProjection.x + 20} y={queryProjection.y - 16}>Query</text>
            </g>
          </g>
        </svg>
        <strong>{shorten(query, 72)}</strong>
      </div>

      <div className="score-board">
        <div className="retrieve-step-title">
          <span>1. cosine similarity</span>
          <strong>방향이 비슷할수록 점수가 높습니다</strong>
        </div>
        {candidates.slice(0, 5).map((match, index) => {
          const score = Number(match.score) || 0
          return (
            <article className="score-row" key={`score-${match.chunk_id ?? index}`}>
              <span>{compactId(match.chunk_id ?? `chunk_${index + 1}`)}</span>
              <div className="score-track">
                <i style={{ width: `${Math.round((score / maxScore) * 100)}%` }} />
              </div>
              <strong>{score.toFixed(4)}</strong>
            </article>
          )
        })}
        <p>점수를 계산한 뒤 높은 순서로 정렬합니다.</p>
      </div>

      <div className="ranking-board">
        <div className="retrieve-step-title">
          <span>2. top-k 선택</span>
          <strong>가장 비슷한 {topK}개 chunk</strong>
        </div>
        {selectedMatches.slice(0, topK).map((match, index) => {
          const score = Number(match.score) || 0
          return (
            <article className="rank-row" key={match.chunk_id ?? index}>
              <span className="rank-index">#{index + 1}</span>
              <div>
                <strong>{match.chunk_id ?? `match_${index + 1}`}</strong>
                <p>{shorten(match.text, 70)}</p>
              </div>
              <span className="score-value">{score.toFixed(4)}</span>
            </article>
          )
        })}
      </div>
    </div>
  )
}

function StoreView({ stage }) {
  const records = normalizeStoreRecords(stage.rawInput ?? stage.input)
  const output = typeof stage.rawOutput === 'object' ? stage.rawOutput : {}
  const collection = output.collection ?? 'vector_db'
  const distance = output.index ?? 'cosine'
  const recordCount = output.vectors ?? records.length

  return (
    <div className="store-viz">
      <div className="store-records">
        <div className="store-step-label">
          <span>1. vector record</span>
          <strong>embedding + metadata</strong>
        </div>
        {records.slice(0, 5).map((record, index) => (
          <article className="store-record" key={`${record.chunkId}-${index}`}>
            <span>{record.chunkId}</span>
            <strong>{record.metadata}</strong>
            <div className="record-parts" aria-label="stored vector record fields">
              <i>embedding vector</i>
              <i>chunk text</i>
              <i>metadata</i>
            </div>
          </article>
        ))}
      </div>
      <div className="store-arrow">
        <span>2. upsert</span>
        <strong>save records</strong>
      </div>
      <div className="collection-box">
        <span>3. Chroma collection</span>
        <strong>{collection}</strong>
        <div className="collection-stats" aria-label="stored collection summary">
          <article>
            <span>records</span>
            <strong>{recordCount}</strong>
          </article>
          <article>
            <span>distance</span>
            <strong>{distance}</strong>
          </article>
          <article>
            <span>lookup key</span>
            <strong>chunk id</strong>
          </article>
        </div>
        <p>인덱스 내부 구조는 Chroma가 관리합니다. 데모에서는 저장된 벡터와 metadata가 검색 단계에서 다시 조회된다는 점만 보여줍니다.</p>
      </div>
    </div>
  )
}

function PromptView({ prompt }) {
  const contexts = asArray(prompt?.contexts)

  return (
    <div className="prompt-assembly">
      <article className="prompt-ingredient">
        <span>system rule</span>
        <strong>근거만 사용, 출처 포함</strong>
      </article>
      <div className="prompt-plus">+</div>
      <article className="prompt-ingredient">
        <span>user query</span>
        <strong>{prompt?.query ?? '사용자 질문'}</strong>
      </article>
      <div className="prompt-plus">+</div>
      <article className="prompt-ingredient">
        <span>retrieved evidence</span>
        <strong>{contexts.length} chunks</strong>
        <p>{shorten(contexts[0], 96)}</p>
      </article>
      <div className="prompt-plus">=</div>
      <article className="prompt-final">
        <span>final prompt</span>
        <strong>Gemini에게 전달되는 최종 입력</strong>
        <p>{shorten(prompt?.prompt_preview ?? 'System + Query + Evidence', 120)}</p>
      </article>
    </div>
  )
}

function GenerateView({ stage }) {
  const result = stage.rawOutput ?? stage.output
  const prompt = stage.rawInput ?? {}
  const answer = result?.answer ?? '답변이 생성되면 여기에 표시됩니다.'

  return (
    <div className="generate-flow">
      <article className="generate-card prompt-packet">
        <span>final prompt</span>
        <strong>{shorten(prompt?.query ?? 'query + evidence', 64)}</strong>
        <p>{asArray(prompt?.contexts).length || 0} retrieved contexts attached</p>
      </article>
      <div className="model-core">
        <span>{result?.model ?? 'Gemini'}</span>
        <strong>grounded generation</strong>
      </div>
      <article className="generate-card answer-card">
        <span>answer + source</span>
        <strong>근거 기반 답변</strong>
        <p>{answer}</p>
      </article>
    </div>
  )
}

function FallbackView({ stage }) {
  return (
    <div className="fallback-viz">
      <span>{stage.payload}</span>
      <strong>{stage.title}</strong>
    </div>
  )
}

function asArray(value) {
  if (Array.isArray(value)) return value
  if (value === null || value === undefined) return []
  return [value]
}

function normalizeVectors(vectors) {
  const samples = [
    [0.12, -0.31, 0.77, -0.08],
    [0.08, -0.24, 0.69, 0.18],
    [-0.14, 0.27, 0.44, -0.35],
  ]
  const topics = ['환불 정책', '배송 예외', '고객 지원']

  return asArray(vectors).map((vector, index) => {
    const fallback = fallbackPoint(index)
    if (typeof vector === 'object' && vector?.embedding_preview) {
      return {
        ...vector,
        label: compactId(vector.id),
        topic: inferTopic(vector.text ?? vector.id ?? '', index),
        point3d: vector.point3d ?? vectorPoint3d(vector, index),
      }
    }

    const text = String(vector ?? '')
    return {
      id: typeof vector === 'string' ? text.split('\n')[0] : `vector_${index + 1}`,
      label: typeof vector === 'string' ? compactId(text.split('\n')[0]) : `vector_${index + 1}`,
      topic: inferTopic(text, index) || topics[index % topics.length],
      text,
      embedding_preview: samples[index % samples.length],
      point3d: vectorPoint3d({ embedding_preview: samples[index % samples.length] }, index),
    }
  })
}

function fallbackPoint(index) {
  const angle = index * 2.399963229728653
  const radius = 12 + (index % 6) * 6
  return {
    x: clamp(50 + Math.cos(angle) * radius, 8, 92),
    y: clamp(50 + Math.sin(angle) * radius, 14, 86),
  }
}

function rankedPoint3d(index) {
  const points = [
    { x: 11, y: 9, z: 13 },
    { x: -15, y: -7, z: 8 },
    { x: 22, y: -18, z: -3 },
    { x: -28, y: 18, z: -10 },
    { x: 34, y: 22, z: 4 },
  ]
  return points[index % points.length]
}

function vectorPoint3d(vector, index) {
  const preview = vector.embedding_preview ?? []
  const fallback = fallbackPoint(index)
  return {
    x: clamp(Number(preview[0]) * 54 || fallback.x - 50, -42, 42),
    y: clamp(Number(preview[1]) * -54 || 50 - fallback.y, -34, 34),
    z: clamp(Number(preview[2]) * 46 || ((index % 7) - 3) * 7, -32, 32),
  }
}

function projectPoint(point) {
  const x = 50 + point.x * 0.78 + point.z * 0.32
  const y = 54 - point.y * 0.68 + point.z * 0.24
  const depth = clamp((point.z + 36) / 72, 0, 1)
  return {
    x: clamp(x, 8, 92),
    y: clamp(y, 10, 90),
    scale: (0.9 + depth * 0.46).toFixed(3),
    opacity: (0.64 + depth * 0.36).toFixed(3),
  }
}

function projectPlotPoint(point) {
  const depth = clamp((point.z + 42) / 84, 0, 1)
  return {
    x: 600 + point.x * 6.8 + point.z * 4.2,
    y: 388 - point.y * 4.8 - point.z * 2.7,
    depth,
  }
}

function clusterIndex(point, index) {
  if (point.x > 8 && point.y > 0) return 1
  if (point.x < -8 && point.y > 0) return 2
  if (point.z > 8) return 3
  return index % 4
}

function compactId(value) {
  return String(value ?? '')
    .replace(/^upload_[^_]+_/, '')
    .replace(/_chunk_/, ' #')
}

function inferTopic(text, index) {
  if (text.includes('환불')) return '환불 정책'
  if (text.includes('배송')) return '배송 예외'
  if (text.includes('고객') || text.includes('지원')) return '고객 지원'
  if (text.includes('DFS') || text.includes('Depth')) return 'DFS 탐색'
  if (text.includes('path')) return '경로 탐색'
  return ['문서 개요', '핵심 근거', '보조 근거'][index % 3]
}

function normalizeMatches(matches) {
  const fallbackScores = [0.91, 0.84, 0.78]

  return asArray(matches).map((match, index) => {
    if (typeof match === 'object' && match?.score !== undefined) {
      return match
    }

    const text = String(match ?? '')
    const scoreMatch = text.match(/score:\s*([0-9.]+)/i)
    return {
      chunk_id: text.split('\n')[0] || `chunk_${index + 1}`,
      score: scoreMatch ? Number(scoreMatch[1]) : fallbackScores[index % fallbackScores.length],
      text,
    }
  })
}

function normalizeRetrieval(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const candidates = normalizeMatches(value.candidates)
    const topK = normalizeMatches(value.top_k)
    return {
      candidates: candidates.length ? candidates : topK,
      topK: topK.length ? topK : candidates,
      queryPoint3d: value.query_point3d ?? value.queryPoint3d,
    }
  }

  const matches = normalizeMatches(value)
  return { candidates: matches, topK: matches, queryPoint3d: { x: 0, y: 0, z: 0 } }
}

function normalizeStoreRecords(value) {
  const items = asArray(value)
  const fallback = ['chunk_001', 'chunk_002', 'chunk_003', 'chunk_004', 'chunk_005']

  return (items.length ? items : fallback).map((item, index) => {
    if (typeof item === 'object') {
      return {
        chunkId: compactId(item.chunk_id ?? item.id ?? `chunk_${index + 1}`),
        metadata: item.metadata ?? item.filename ?? 'metadata',
      }
    }

    const text = String(item)
    return {
      chunkId: compactId(text.split('\n')[0] || `chunk_${index + 1}`),
      metadata: text.includes('metadata') ? 'metadata linked' : 'source metadata',
    }
  })
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function shorten(value, maxLength) {
  const text = String(value ?? '')
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength - 1)}...`
}

function estimateTokens(value) {
  const text = String(value ?? '').trim()
  if (!text) return 0
  return Math.max(1, Math.round(text.length / 4))
}

export default StageFocus

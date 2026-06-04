import { useEffect, useState } from 'react'

function ResultPanel({ answer, sources, modelName }) {
  const [visibleAnswer, setVisibleAnswer] = useState('')

  useEffect(() => {
    if (!answer) {
      setVisibleAnswer('')
      return undefined
    }

    setVisibleAnswer('')
    let index = 0
    const timer = window.setInterval(() => {
      index += 2
      setVisibleAnswer(answer.slice(0, index))
      if (index >= answer.length) window.clearInterval(timer)
    }, 16)

    return () => window.clearInterval(timer)
  }, [answer])

  if (!answer && sources.length === 0) return null

  return (
    <section className="result-panel">
      <div>
        <span>Answer {modelName ? `· ${modelName}` : ''}</span>
        <p className="typed-answer">{visibleAnswer}</p>
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
  )
}

export default ResultPanel

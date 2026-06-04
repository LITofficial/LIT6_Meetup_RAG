function StageList({ stages, activeStage, onSelectStage }) {
  return (
    <aside className="stage-list" aria-label="RAG stages">
      {stages.map((stage, index) => (
        <button
          key={stage.id}
          type="button"
          className={index === activeStage ? 'stage-tab active' : 'stage-tab'}
          onClick={() => onSelectStage(index)}
        >
          <span className="stage-index">{String(index + 1).padStart(2, '0')}</span>
          <span>
            <strong>{stage.title}</strong>
            <small>{stage.label}</small>
          </span>
        </button>
      ))}
    </aside>
  )
}

export default StageList

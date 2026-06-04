function Controls({ stages, activeStage, onMoveStage, onSelectStage }) {
  return (
    <div className="controls">
      <button type="button" onClick={() => onMoveStage(-1)} aria-label="previous stage">
        ←
      </button>
      <div className="step-dots" aria-label="stage position">
        {stages.map((stage, index) => (
          <button
            key={stage.id}
            type="button"
            className={index === activeStage ? 'dot active' : 'dot'}
            onClick={() => onSelectStage(index)}
            aria-label={stage.title}
          />
        ))}
      </div>
      <button type="button" onClick={() => onMoveStage(1)} aria-label="next stage">
        →
      </button>
    </div>
  )
}

export default Controls

function PipelineMap({ stages, activeStage, connections, onSelectStage }) {
  return (
    <div className="pipeline-map">
      {stages.map((stage, index) => (
        <div key={stage.id} className="map-segment">
          <button
            type="button"
            className={index === activeStage ? 'node active' : 'node'}
            onClick={() => onSelectStage(index)}
            aria-label={stage.title}
          >
            <span>{stage.label}</span>
            <strong>{stage.payload}</strong>
          </button>
          {index < stages.length - 1 && (
            <div className={index < activeStage ? 'connector filled' : 'connector'}>
              <span>{connections[index]}</span>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

export default PipelineMap

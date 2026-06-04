function DetailPanel({ label, title, stageId, items }) {
  return (
    <section className="detail-panel">
      <div className="panel-heading">
        <span>{label}</span>
        <strong>{title}</strong>
      </div>
      <ul>
        {items.map((item, index) => (
          <li key={`${stageId}-${label.toLowerCase()}-${index}`}>{item}</li>
        ))}
      </ul>
    </section>
  )
}

export default DetailPanel

import { useState, useRef, useEffect } from 'react'

const VIDEO_EXTS = new Set(['.mp4', '.mov'])
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.gif', '.webp'])

function getExt(filename) {
  return filename.slice(filename.lastIndexOf('.')).toLowerCase()
}

function isVideo(filename) {
  return VIDEO_EXTS.has(getExt(filename))
}

function ScoreBar({ score }) {
  return (
    <div className="score-bar-wrap">
      <div className="score-bar" style={{ width: `${(score * 100).toFixed(1)}%` }} />
      <span className="score-label">{(score * 100).toFixed(2)}%</span>
    </div>
  )
}

function VideoPreview({ filename }) {
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const debounceRef = useRef(null)
  const [frameSrc, setFrameSrc] = useState(
    `/api/frame/${encodeURIComponent(filename)}?t=0`
  )

  useEffect(() => {
    fetch(`/api/video-duration/${encodeURIComponent(filename)}`)
      .then(r => r.json())
      .then(d => setDuration(d.duration || 0))
      .catch(() => {})
  }, [filename])

  function handleSliderChange(e) {
    const t = parseFloat(e.target.value)
    setCurrentTime(t)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setFrameSrc(`/api/frame/${encodeURIComponent(filename)}?t=${t}`)
    }, 150)
  }

  function formatTime(s) {
    if (!isFinite(s)) return '0:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60).toString().padStart(2, '0')
    return `${m}:${sec}`
  }

  return (
    <div className="video-wrap">
      <img src={frameSrc} alt={`${filename} at ${formatTime(currentTime)}`} className="media-preview" />
      <div className="seek-controls">
        <span className="time-display">{formatTime(currentTime)}</span>
        <input
          type="range"
          className="seek-slider"
          min={0}
          max={duration || 100}
          step={0.1}
          value={currentTime}
          onChange={handleSliderChange}
        />
        <span className="time-display">{formatTime(duration)}</span>
      </div>
    </div>
  )
}

function MediaPreview({ filename }) {
  if (isVideo(filename)) {
    return <VideoPreview filename={filename} />
  }
  const src = `/sources/${encodeURIComponent(filename)}`
  return <img src={src} alt={filename} className="media-preview" />
}

function HistoryItem({ entry, rank }) {
  const [expanded, setExpanded] = useState(false)
  const top = entry.results[0]
  const rest = entry.results.slice(1)

  return (
    <div className="history-card">
      <div className="card-header">
        <span className="rank-badge">#{rank}</span>
        <div className="prompt-text">"{entry.prompt}"</div>
        <span className="timestamp">{entry.time}</span>
      </div>

      <div className="card-body">
        <div className="top-result">
          <div className="result-meta">
            <span className="result-rank">Top Match</span>
            <span className="result-filename">{top.name}</span>
            <ScoreBar score={top.score} />
          </div>
          <MediaPreview filename={top.name} />
        </div>

        {rest.length > 0 && (
          <div className="other-results">
            <button className="toggle-btn" onClick={() => setExpanded(x => !x)}>
              {expanded ? '▲ Hide' : '▼ Show'} other {rest.length} matches
            </button>
            {expanded && (
              <div className="other-grid">
                {rest.map((r, i) => (
                  <div key={r.name} className="other-result">
                    <div className="result-meta small">
                      <span className="result-rank">#{i + 2}</span>
                      <span className="result-filename">{r.name}</span>
                      <ScoreBar score={r.score} />
                    </div>
                    <MediaPreview filename={r.name} />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  async function handleSearch() {
    if (!prompt.trim() || loading) return
    setLoading(true)
    setError('')

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), top_k: 5 }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Search failed')

      setHistory(prev => [
        {
          id: Date.now(),
          prompt: prompt.trim(),
          results: data.results,
          time: new Date().toLocaleTimeString(),
        },
        ...prev,
      ])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSearch()
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-title">
          <span className="gem-icon">◆</span>
          <h1>Gemini Multimodal Embedding Search</h1>
        </div>
        <p className="header-sub">Search your media library by describing what you see</p>
      </header>

      <div className="search-section">
        <div className="input-wrap">
          <textarea
            ref={textareaRef}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe what you're looking for… e.g. &quot;a dog running in the park&quot;"
            className="prompt-input"
            rows={3}
            disabled={loading}
          />
          <button
            onClick={handleSearch}
            disabled={!prompt.trim() || loading}
            className="search-btn"
          >
            {loading ? (
              <span className="spinner" />
            ) : (
              <>
                <span>Search</span>
                <span className="hint">Ctrl+↵</span>
              </>
            )}
          </button>
        </div>
        {error && <div className="error-msg">⚠ {error}</div>}
      </div>

      {history.length > 0 && (
        <section className="history-section">
          <h2 className="history-heading">
            Search History <span className="count">({history.length})</span>
          </h2>
          <div className="history-grid">
            {history.map((entry, i) => (
              <HistoryItem key={entry.id} entry={entry} rank={history.length - i} />
            ))}
          </div>
        </section>
      )}

      {history.length === 0 && !loading && (
        <div className="empty-state">
          <div className="empty-icon">🔍</div>
          <p>Enter a prompt above to search your embedded media files</p>
        </div>
      )}
    </div>
  )
}

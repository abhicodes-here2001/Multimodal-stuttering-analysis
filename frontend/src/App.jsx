import React, { useState, useRef } from 'react';
import axios from 'axios';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, 
  AreaChart, Area, XAxis, YAxis, Tooltip, Legend,
  CartesianGrid, BarChart, Bar
} from 'recharts';
import './App.css';

const API_URL = 'http://localhost:8000';

const STUTTER_COLORS = {
  Prolongation: '#ef4444',
  Block: '#f97316', 
  SoundRep: '#eab308',
  WordRep: '#8b5cf6',
  Interjection: '#3b82f6'
};

const SEVERITY_COLORS = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'];

function App() {
  const [file, setFile] = useState(null);
  const [patientName, setPatientName] = useState('');
  const [patientId, setPatientId] = useState('');
  const [clinicianName, setClinicianName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [report, setReport] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const fileRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return setError('Please select an audio file');

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_name', patientName || 'Anonymous');
    formData.append('patient_id', patientId || '');
    formData.append('clinician_name', clinicianName || '');
    formData.append('threshold', 0.4);

    try {
      const res = await axios.post(`${API_URL}/analyze`, formData);
      setReport(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  // Upload Page
  if (!report) {
    return (
      <div className="upload-page">
        <div className="upload-box">
          <div className="upload-header">
            <div className="logo">🎙️</div>
            <h1>Speech Fluency Analysis</h1>
            <p>Clinical stutter detection powered by AI</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div 
              className={`dropzone ${file ? 'has-file' : ''}`}
              onClick={() => fileRef.current?.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".wav,.mp3,.m4a,.flac"
                onChange={(e) => { setFile(e.target.files[0]); setError(null); }}
                hidden
              />
              {file ? (
                <div className="file-selected">
                  <span className="file-icon">🎵</span>
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">{(file.size/1024/1024).toFixed(1)} MB</span>
                </div>
              ) : (
                <>
                  <span className="drop-icon">📁</span>
                  <span>Click or drag audio file</span>
                  <span className="formats">WAV, MP3, M4A, FLAC</span>
                </>
              )}
            </div>

            <div className="form-row">
              <input 
                type="text" 
                placeholder="Patient Name" 
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
              />
              <input 
                type="text" 
                placeholder="Patient ID" 
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
              />
            </div>
            <input 
              type="text" 
              placeholder="Clinician Name" 
              value={clinicianName}
              onChange={(e) => setClinicianName(e.target.value)}
            />

            {error && <div className="error">{error}</div>}

            <button type="submit" disabled={loading || !file}>
              {loading ? 'Analyzing...' : 'Analyze Speech'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Report Page
  const { metrics, stutter_analysis, chart_data, recommendations } = report;
  const wordLevel = stutter_analysis?.word_level || [];
  const timeline = stutter_analysis?.timeline || [];
  const typeData = chart_data?.type_distribution || [];

  return (
    <div className="report-page">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <span className="header-logo">🎙️</span>
          <div>
            <h1>Stutter Analysis Report</h1>
            <span className="report-id">{report.report_id}</span>
          </div>
        </div>
        <div className="header-actions">
          <button className="btn-download" onClick={() => {
            const dataStr = JSON.stringify(report, null, 2);
            const blob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${report.report_id}.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}>
            📥 Download Report
          </button>
          <button className="btn-new" onClick={() => { setReport(null); setFile(null); }}>
            + New Analysis
          </button>
        </div>
      </header>

      {/* Patient Bar */}
      <div className="patient-bar">
        <span><strong>Patient:</strong> {report.patient_info?.name}</span>
        <span><strong>ID:</strong> {report.patient_info?.id || 'N/A'}</span>
        <span><strong>Clinician:</strong> {report.clinician || 'N/A'}</span>
        <span><strong>Date:</strong> {new Date(report.generated_at).toLocaleDateString()}</span>
      </div>

      {/* Tabs */}
      <nav className="tabs">
        {['overview', 'transcription', 'timeline', 'details'].map(tab => (
          <button 
            key={tab}
            className={activeTab === tab ? 'active' : ''}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="content">
        {activeTab === 'overview' && (
          <div className="overview">
            {/* Top Row: Severity + Metrics */}
            <div className="top-row">
              <div className="severity-card">
                <h3>Severity Score</h3>
                <div className="severity-display">
                  <div 
                    className="severity-circle"
                    style={{ borderColor: SEVERITY_COLORS[metrics.severity_score - 1] }}
                  >
                    <span className="score">{metrics.severity_score}</span>
                    <span className="max">/5</span>
                  </div>
                  <span 
                    className="severity-label"
                    style={{ color: SEVERITY_COLORS[metrics.severity_score - 1] }}
                  >
                    {metrics.severity_label}
                  </span>
                </div>
              </div>

              <div className="metrics-row">
                <div className="metric">
                  <span className="metric-value">{metrics.chunk_stutter_rate}%</span>
                  <span className="metric-label">Stutter Rate</span>
                </div>
                <div className="metric">
                  <span className="metric-value">{metrics.words_with_stutter}/{metrics.total_words}</span>
                  <span className="metric-label">Words Affected</span>
                </div>
                <div className="metric">
                  <span className="metric-value">{metrics.total_duration_sec}s</span>
                  <span className="metric-label">Duration</span>
                </div>
                <div className="metric">
                  <span className="metric-value">{metrics.speaking_rate_wpm}</span>
                  <span className="metric-label">Words/Min</span>
                </div>
              </div>
            </div>

            {/* Charts Row */}
            <div className="charts-row">
              <div className="chart-card">
                <h3>Stutter Types</h3>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={180}>
                    <PieChart>
                      <Pie
                        data={typeData.filter(d => d.value > 0)}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={70}
                        dataKey="value"
                      >
                        {typeData.map((entry, i) => (
                          <Cell key={i} fill={STUTTER_COLORS[entry.name]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="chart-legend">
                    {typeData.map((t, i) => (
                      <div key={i} className="legend-item">
                        <span className="dot" style={{ background: STUTTER_COLORS[t.name] }}></span>
                        <span>{t.name}</span>
                        <span className="count">{t.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="chart-card">
                <h3>Timeline Pattern</h3>
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={timeline.map(t => ({ time: t.time.toFixed(1), count: t.stutter_count }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Area type="monotone" dataKey="count" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recommendations */}
            <div className="recommendations">
              <h3>Clinical Recommendations</h3>
              <ul>
                {recommendations?.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'transcription' && (
          <div className="transcription">
            <h3>Annotated Transcription</h3>
            <p className="hint">Hover over highlighted words to see stutter details</p>
            
            <div className="transcript-box">
              {wordLevel.map((w, i) => (
                <span 
                  key={i}
                  className={`word ${w.has_stutter ? 'stuttered' : ''}`}
                  style={w.has_stutter ? { 
                    borderBottomColor: STUTTER_COLORS[w.stutters[0]],
                    backgroundColor: `${STUTTER_COLORS[w.stutters[0]]}15`
                  } : {}}
                  title={w.has_stutter ? `${w.stutters.join(', ')} (${w.start.toFixed(2)}s)` : ''}
                >
                  {w.word}
                  {w.has_stutter && (
                    <span className="stutter-tag" style={{ background: STUTTER_COLORS[w.stutters[0]] }}>
                      {w.stutters[0].slice(0, 3)}
                    </span>
                  )}
                </span>
              ))}
            </div>

            <div className="legend-bar">
              {Object.entries(STUTTER_COLORS).map(([type, color]) => (
                <span key={type} className="legend-chip">
                  <span className="chip-dot" style={{ background: color }}></span>
                  {type}
                </span>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'timeline' && (
          <div className="timeline-view">
            <h3>Stutter Pattern Over Time</h3>
            
            <div className="big-chart">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={timeline.map(t => {
                  const d = { time: `${t.time.toFixed(1)}s` };
                  Object.keys(STUTTER_COLORS).forEach(type => {
                    d[type] = t.types.includes(type) ? 1 : 0;
                  });
                  return d;
                })}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {Object.entries(STUTTER_COLORS).map(([type, color]) => (
                    <Bar key={type} dataKey={type} stackId="a" fill={color} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            <h3>Probability Heatmap</h3>
            <div className="heatmap">
              <div className="heatmap-header">
                <span>Time</span>
                {Object.keys(STUTTER_COLORS).map(t => <span key={t}>{t}</span>)}
              </div>
              {timeline.slice(0, 20).map((seg, i) => (
                <div key={i} className="heatmap-row">
                  <span className="time-cell">{seg.time.toFixed(1)}s</span>
                  {Object.keys(STUTTER_COLORS).map(type => {
                    const prob = seg.probabilities?.[type] || 0;
                    return (
                      <span 
                        key={type} 
                        className="prob-cell"
                        style={{ 
                          background: `rgba(${type === 'Prolongation' ? '239,68,68' : 
                            type === 'Block' ? '249,115,22' : 
                            type === 'SoundRep' ? '234,179,8' : 
                            type === 'WordRep' ? '139,92,246' : '59,130,246'}, ${prob})`,
                          color: prob > 0.5 ? '#fff' : '#374151'
                        }}
                      >
                        {(prob * 100).toFixed(0)}%
                      </span>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'details' && (
          <div className="details-view">
            <div className="tables-grid">
              <div className="table-section">
                <h3>Word Analysis</h3>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Word</th>
                        <th>Time</th>
                        <th>Stutter</th>
                      </tr>
                    </thead>
                    <tbody>
                      {wordLevel.map((w, i) => (
                        <tr key={i} className={w.has_stutter ? 'row-stutter' : ''}>
                          <td>{i + 1}</td>
                          <td><strong>{w.word}</strong></td>
                          <td>{w.start.toFixed(2)}s</td>
                          <td>
                            {w.stutters.length > 0 ? w.stutters.map((s, j) => (
                              <span key={j} className="tag" style={{ background: STUTTER_COLORS[s] }}>{s}</span>
                            )) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="table-section">
                <h3>Segment Analysis</h3>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Detected</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {timeline.map((seg, i) => (
                        <tr key={i} className={seg.stutter_count > 0 ? 'row-stutter' : ''}>
                          <td>{seg.start.toFixed(2)}s - {seg.end.toFixed(2)}s</td>
                          <td>
                            {seg.types.length > 0 ? seg.types.map((s, j) => (
                              <span key={j} className="tag" style={{ background: STUTTER_COLORS[s] }}>{s}</span>
                            )) : <span className="clear">Clear</span>}
                          </td>
                          <td>
                            {seg.types.length > 0 && seg.types.map((s, j) => (
                              <span key={j} className="conf">{(seg.probabilities[s] * 100).toFixed(0)}%</span>
                            ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p><strong>Disclaimer:</strong> AI-assisted analysis for clinical support only. Consult a qualified SLP for diagnosis.</p>
      </footer>
    </div>
  );
}

export default App;

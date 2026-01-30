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
const SEVERITY_NAMES = ['Very Mild', 'Mild', 'Moderate', 'Severe', 'Very Severe'];

function App() {
  const [file, setFile] = useState(null);
  const [patientName, setPatientName] = useState('');
  const [patientId, setPatientId] = useState('');
  const [clinicianName, setClinicianName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [report, setReport] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [threshold, setThreshold] = useState(0.4);
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
    formData.append('threshold', threshold);

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
        <div className="upload-box" style={{ position: 'relative' }}>
          {loading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <p className="loading-text">Analyzing Audio...</p>
              <p className="loading-subtext">This may take a minute or two depending on file size.</p>
            </div>
          )}

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
                placeholder="Patient ID (Optional)" 
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
              />
            </div>
            <input 
              type="text" 
              placeholder="Clinician Name (Optional)" 
              value={clinicianName}
              onChange={(e) => setClinicianName(e.target.value)}
            />

            <div className="threshold-slider">
              <label>Detection Threshold: <span>{threshold.toFixed(2)}</span></label>
              <p>Lower is more sensitive, higher is more conservative.</p>
              <input 
                type="range" 
                min="0.1" 
                max="0.9" 
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
            </div>

            <button type="submit" disabled={loading || !file}>
              {loading ? 'Analyzing...' : 'Analyze Speech'}
            </button>
            {error && <p className="error-message">{error}</p>}
          </form>
        </div>
      </div>
    );
  }

  // Report Page
  const { analysis, metadata } = report;
  
  // Extract detailed analysis data that might be nested
  const wordLevel = report.stutter_analysis?.word_level || [];
  const timeline = report.stutter_analysis?.timeline || report.chart_data?.timeline || [];

  const pieData = Object.entries(analysis.stutter_type_distribution)
    .map(([name, value]) => ({ name, value }))
    .filter(d => d.value > 0);

  const severity = analysis.severity_assessment;

  return (
    <div className="report-page">
      <header className="report-header">
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

      <nav className="report-nav">
        <div className="nav-item">
          <span>Patient:</span>
          <strong>{report.patient_info?.name}</strong>
        </div>
        <div className="nav-item">
          <span>ID:</span>
          <strong>{report.patient_info?.id || 'N/A'}</strong>
        </div>
        <div className="nav-item">
          <span>Clinician:</span>
          <strong>{report.clinician || 'N/A'}</strong>
        </div>
        <div className="nav-item">
          <span>Date:</span>
          <strong>{new Date(report.generated_at).toLocaleDateString()}</strong>
        </div>
      </nav>

      <div className="tabs">
        {['overview', 'transcription', 'timeline', 'details'].map(tab => (
          <button 
            key={tab}
            className={activeTab === tab ? 'active' : ''}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <main className="report-content">
        {activeTab === 'overview' && (
          <div className="overview">
            <div className="overview-header">
              <h2>Analysis Overview</h2>
            </div>

            <div className="overview-grid">
              <div className="grid-item chart-container">
                <h3>Stutter Type Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={STUTTER_COLORS[entry.name]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="grid-item severity-meter">
                <h3>Overall Severity</h3>
                <div className="meter-container">
                  <div className="meter-bar">
                    {SEVERITY_NAMES.map((name, index) => (
                      <div 
                        key={name}
                        className={`meter-segment ${severity.level_value === index + 1 ? 'active' : ''}`}
                        style={{ backgroundColor: SEVERITY_COLORS[index] }}
                      ></div>
                    ))}
                  </div>
                  <div 
                    className="meter-indicator" 
                    style={{ left: `${((severity.level_value - 0.5) / 5) * 100}%` }}
                  >
                    <div className="indicator-head"></div>
                    <div className="indicator-line"></div>
                  </div>
                </div>
                <p className="severity-label" style={{ color: SEVERITY_COLORS[severity.level_value - 1] }}>
                  {severity.level_name}
                </p>
                <p className="severity-detail">Based on a word stutter rate of {severity.word_stutter_rate.toFixed(2)}%</p>
              </div>

              <div className="grid-item transcript-preview">
                <h3>Transcription Highlight</h3>
                <div className="transcript-sample">
                  {report.transcription_sample}
                </div>
              </div>
            </div>

            <div className="overview-details">
              <h3>Analysis Details</h3>
              <div className="details-item">
                <span className="detail-label">Total Words:</span>
                <span className="detail-value">{analysis.total_words}</span>
              </div>
              <div className="details-item">
                <span className="detail-label">Fluent Words:</span>
                <span className="detail-value">{analysis.fluent_words}</span>
              </div>
              <div className="details-item">
                <span className="detail-label">Disfluent Words:</span>
                <span className="detail-value">{analysis.disfluent_words}</span>
              </div>
              <div className="details-item">
                <span className="detail-label">Prolongations:</span>
                <span className="detail-value">{analysis.prolongations}</span>
              </div>
              <div className="details-item">
                <span className="detail-label">Blocks:</span>
                <span className="detail-value">{analysis.blocks}</span>
              </div>
              <div className="details-item">
                <span className="detail-label">Sound Repetitions:</span>
                <span className="detail-value">{analysis.sound_repetitions}</span>
              </div>
              <div className="details-item">
                <span className="detail-label">Word Repetitions:</span>
                <span className="detail-value">{analysis.word_repetitions}</span>
              </div>
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

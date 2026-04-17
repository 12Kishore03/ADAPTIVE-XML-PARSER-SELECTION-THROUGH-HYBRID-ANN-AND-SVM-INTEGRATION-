import React, { useState, useEffect } from 'react';
import './App.css';

const BACKEND_URL = "https://xml-parser-backend.onrender.com";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [cpuCores, setCpuCores] = useState(4);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/health`);
      const health = await response.json();
      
      if (health.models_loaded) {
        setBackendStatus('ml_healthy');
      } else if (health.status === 'healthy') {
        setBackendStatus('rule_healthy');
      } else {
        setBackendStatus('unreachable');
      }
    } catch (err) {
      setBackendStatus('unreachable');
      setError('Backend server is unreachable. Please wait (Render may be waking up)');
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.name.endsWith('.xml')) {
        setSelectedFile(file);
        setError('');
        setPrediction(null);
      } else {
        setError('Please select a valid XML file');
        event.target.value = '';
      }
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.name.endsWith('.xml')) {
      setSelectedFile(file);
      setError('');
      setPrediction(null);
    } else {
      setError('Please drop a valid XML file');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an XML file first');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('xml_file', selectedFile);
      formData.append('cpu_cores', cpuCores.toString());

      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadParserCode = () => {
    if (!prediction?.parser_code) return;
    
    const blob = new Blob([prediction.parser_code], { type: 'text/python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${prediction.algorithm.toLowerCase()}_parser.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getStatusInfo = () => {
    switch (backendStatus) {
      case 'ml_healthy':
        return { text: 'ML Backend Connected ', color: '#10b981', icon: '' };
      case 'rule_healthy':
        return { text: 'Rule-Based Backend Connected 📋', color: '#f59e0b', icon: '⚡' };
      case 'unreachable':
        return { text: 'Backend Unreachable ', color: '#ef4444', icon: '⚠️' };
      default:
        return { text: 'Checking...', color: '#6b7280', icon: '🔍' };
    }
  };

  const statusInfo = getStatusInfo();

  return (
    <div className="app">
      <header className="app-header">
        <h1> XML Parser Predictor</h1>
        <p>AI-powered XML parsing algorithm recommendation system</p>
        <div className="status-indicator">
          <div 
            className="status-dot"
            style={{ backgroundColor: statusInfo.color }}
          ></div>
          <span>{statusInfo.icon} {statusInfo.text}</span>
        </div>
      </header>

      <main className="app-main">
        <div className="input-section">
          <div className="card">
            <h2>📁 Upload XML File</h2>
            <div 
              className="file-upload-area"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <input
                id="file-input"
                type="file"
                accept=".xml"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <div className="upload-content">
                <div className="upload-icon">📄</div>
                <p>Click to select or drag & drop XML file</p>
                <small>Maximum file size: 200MB</small>
              </div>
            </div>
            {selectedFile && (
              <div className="file-info">
                <strong>Selected:</strong> {selectedFile.name} 
                ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </div>
            )}
          </div>

          <div className="card">
            <h2>⚙️ CPU Configuration</h2>
            <div className="core-selector">
              <div className="core-options">
                {[1, 2, 4, 6, 8, 10, 12, 14, 16].map(cores => (
                  <button
                    key={cores}
                    className={`core-option ${cpuCores === cores ? 'selected' : ''}`}
                    onClick={() => setCpuCores(cores)}
                  >
                    {cores} Core{cores > 1 ? 's' : ''}
                  </button>
                ))}
              </div>
              <div className="core-info">
                <small>Selected: {cpuCores} CPU Core{cpuCores > 1 ? 's' : ''}</small>
              </div>
            </div>
          </div>

          <div className="predict-button-container">
            <button 
              className="predict-button"
              onClick={handlePredict}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <div className="button-spinner"></div>
                  Analyzing...
                </>
              ) : (
                '🚀 Predict Best Parser'
              )}
            </button>
          </div>
        </div>

        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Analyzing XML structure and predicting optimal parser...</p>
            <p className="loading-subtitle">
              {backendStatus === 'ml_healthy' ? 
                'Using Machine Learning model...' : 
                'Using rule-based analysis...'
              }
            </p>
          </div>
        )}

        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {prediction && (
          <div className="results-section">
            <div className="card main-result">
              <div className={`prediction-header ${prediction.method === 'Machine Learning' ? 'ml-header' : 'rule-header'}`}>
                <h2>
                  {prediction.method === 'Machine Learning' ? '🧠 ML Recommended:' : '📋 Rule-Based Recommended:'} 
                  <span className="algorithm-name"> {prediction.algorithm}</span>
                </h2>
                <span className="method-badge">
                  {prediction.method}
                </span>
              </div>
              
              <div className="confidence-display">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="confidence-text">
                  Confidence: <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
                </span>
              </div>

              <div className="reason">
                <h4>📋 Reasoning:</h4>
                <p>{prediction.reason}</p>
              </div>

              <h3>📊 Extracted Features</h3>
              <div className="features-grid">
                {Object.entries(prediction.extracted_features).map(([key, value]) => (
                  <div key={key} className="feature-item">
                    <span className="feature-name">{key}:</span>
                    <span className="feature-value">{value}</span>
                  </div>
                ))}
              </div>

              {prediction.ml_details && prediction.ml_details.probabilities && (
                <div className="ml-probabilities">
                  <h4>🤖 ML Probability Distribution:</h4>
                  <div className="probabilities-grid">
                    {Object.entries(prediction.ml_details.probabilities).map(([algo, prob]) => (
                      <div key={algo} className="probability-item">
                        <span className="algo-name">{algo}</span>
                        <div className="probability-bar">
                          <div 
                            className="probability-fill"
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                          <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="parser-code">
                <h4>🔧 Parser Implementation Code:</h4>
                <p className="code-description">
                  Save this as a <strong>.py</strong> file and run it with your XML file
                </p>
                <pre className="code-block">
                  <code>{prediction.parser_code}</code>
                </pre>
                <button 
                  onClick={downloadParserCode}
                  className="download-button"
                >
                  💾 Download Parser Code
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="algorithms-section">
          <div className="card">
            <h2>📚 Available Parsing Algorithms</h2>
            <div className="algorithms-grid">
              <div className="algorithm-card" style={{ borderLeftColor: '#3b82f6' }}>
                <h3 style={{ color: '#3b82f6' }}>DOM</h3>
                <p>Best for small files (0-4MB) - Loads entire document into memory</p>
              </div>
              <div className="algorithm-card" style={{ borderLeftColor: '#10b981' }}>
                <h3 style={{ color: '#10b981' }}>JDOM</h3>
                <p>Good for medium files (4-25MB) - Java-like API for Python</p>
              </div>
              <div className="algorithm-card" style={{ borderLeftColor: '#a855f7' }}>
                <h3 style={{ color: '#a855f7' }}>SAX</h3>
                <p>Excellent for large files (25-100MB) - Event-based, memory efficient</p>
              </div>
              <div className="algorithm-card" style={{ borderLeftColor: '#f97316' }}>
                <h3 style={{ color: '#f97316' }}>StAX</h3>
                <p>Optimal for very large files (100+MB) - Pull-based parsing</p>
              </div>
              <div className="algorithm-card" style={{ borderLeftColor: '#ec4899' }}>
                <h3 style={{ color: '#ec4899' }}>PXTG</h3>
                <p>Complex XML structures - Transforms XML to graph representation</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Hybrid Machine Learning Model for Efficient XML Parsing</p>
      </footer>
    </div>
  );
}

export default App;
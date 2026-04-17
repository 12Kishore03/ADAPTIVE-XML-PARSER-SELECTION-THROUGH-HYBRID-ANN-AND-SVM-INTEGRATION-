import React, { useState, useEffect } from 'react';
import './App.css';

// 🔥 ADD THIS (backend URL)
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
      setError('Backend is waking up... please wait a few seconds and try again.');
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
      setError(err.message || 'Something went wrong');
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
        return { text: 'ML Backend Connected', color: '#10b981', icon: '🧠' };
      case 'rule_healthy':
        return { text: 'Rule-Based Backend Connected', color: '#f59e0b', icon: '⚡' };
      case 'unreachable':
        return { text: 'Backend Unreachable', color: '#ef4444', icon: '⚠️' };
      default:
        return { text: 'Checking...', color: '#6b7280', icon: '🔍' };
    }
  };

  const statusInfo = getStatusInfo();

  return (
    <div className="app">
      <header className="app-header">
        <h1>XML Parser Predictor</h1>
        <p>AI-powered XML parsing algorithm recommendation system</p>
        <div className="status-indicator">
          <div className="status-dot" style={{ backgroundColor: statusInfo.color }}></div>
          <span>{statusInfo.icon} {statusInfo.text}</span>
        </div>
      </header>

      <main className="app-main">
        <div className="input-section">
          <div className="card">
            <h2>Upload XML File</h2>
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
              <p>Click or drag XML file here</p>
            </div>

            {selectedFile && (
              <p>Selected: {selectedFile.name}</p>
            )}
          </div>

          <div className="card">
            <h2>CPU Cores</h2>
            {[1, 2, 4, 8].map(c => (
              <button key={c} onClick={() => setCpuCores(c)}>
                {c} cores
              </button>
            ))}
          </div>

          <button onClick={handlePredict} disabled={loading}>
            {loading ? 'Analyzing...' : 'Predict'}
          </button>
        </div>

        {error && <p style={{ color: 'red' }}>{error}</p>}

        {prediction && (
          <div>
            <h2>Recommended: {prediction.algorithm}</h2>
            <p>{prediction.reason}</p>
            <button onClick={downloadParserCode}>Download Code</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
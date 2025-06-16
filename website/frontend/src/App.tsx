import React, { useState } from 'react';
import './App.css';
import Upload from './components/Upload';
import Results from './components/Results';
import { DetectionResult } from './types';

function App() {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Image Detector</h1>
        <p>Upload an image to check if it's AI-generated</p>
      </header>
      
      <main className="App-main">
        <Upload 
          onResult={setResult} 
          onLoading={setLoading}
        />
        
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing image...</p>
          </div>
        )}
        
        {result && !loading && (
          <Results result={result} />
        )}
      </main>
      
      <footer className="App-footer">
        <p>Based on research: "AI-Generated Image Detection Using Advanced Transformer Models"</p>
      </footer>
    </div>
  );
}

export default App;
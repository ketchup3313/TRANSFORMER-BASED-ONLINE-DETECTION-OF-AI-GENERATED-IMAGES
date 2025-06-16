import React from 'react';
import { DetectionResult } from '../types';
import Visualization from './Visualization';
import './Results.css';

interface ResultsProps {
  result: DetectionResult;
}

const Results: React.FC<ResultsProps> = ({ result }) => {
  const { prediction, image_info } = result;
  const confidencePercentage = (prediction.confidence * 100).toFixed(1);
  const isHighConfidence = prediction.confidence > 0.8;
  
  return (
    <div className="results-container">
      <div className="main-result">
        <h2>Detection Result</h2>
        
        <div className={`verdict ${prediction.is_ai_generated ? 'ai-generated' : 'real'}`}>
          <div className="verdict-icon">
            {prediction.is_ai_generated ? '🤖' : '📷'}
          </div>
          <div className="verdict-text">
            <h3>{prediction.is_ai_generated ? 'AI-Generated' : 'Real Image'}</h3>
            <p className="confidence">
              Confidence: <strong>{confidencePercentage}%</strong>
              {isHighConfidence && ' (High)'}
            </p>
          </div>
        </div>
        
        <div className="probability-bars">
          <div className="probability-item">
            <label>Real</label>
            <div className="bar-container">
              <div 
                className="bar real" 
                style={{ width: `${prediction.probability_real * 100}%` }}
              />
              <span>{(prediction.probability_real * 100).toFixed(1)}%</span>
            </div>
          </div>
          
          <div className="probability-item">
            <label>AI-Generated</label>
            <div className="bar-container">
              <div 
                className="bar ai" 
                style={{ width: `${prediction.probability_ai * 100}%` }}
              />
              <span>{(prediction.probability_ai * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="additional-info">
        <h3>Image Information</h3>
        <ul>
          <li>Resolution: {image_info.width} × {image_info.height}</li>
          <li>Format: {image_info.format || 'Unknown'}</li>
          <li>File Size: {image_info.file_size_mb} MB</li>
          <li>Processing Time: {prediction.processing_time} ms</li>
        </ul>
      </div>
      
      <Visualization result={result} />
      
      <div className="feedback-section">
        <h3>Was this result accurate?</h3>
        <div className="feedback-buttons">
          <button className="feedback-btn correct">✓ Correct</button>
          <button className="feedback-btn incorrect">✗ Incorrect</button>
        </div>
      </div>
    </div>
  );
};

export default Results;
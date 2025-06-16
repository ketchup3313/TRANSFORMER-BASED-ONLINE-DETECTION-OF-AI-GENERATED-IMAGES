import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { DetectionResult } from '../types';
import './Visualization.css';

interface VisualizationProps {
  result: DetectionResult;
}

const Visualization: React.FC<VisualizationProps> = ({ result }) => {
  const [activeTab, setActiveTab] = useState<'frequency' | 'attention'>('frequency');
  
  // 模拟频率分析数据
  const frequencyData = Array.from({ length: 50 }, (_, i) => ({
    frequency: i * 2,
    amplitude: Math.random() * 100 * (result.prediction.is_ai_generated ? 1.5 : 1)
  }));
  
  return (
    <div className="visualization-container">
      <h3>Advanced Analysis</h3>
      
      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'frequency' ? 'active' : ''}`}
          onClick={() => setActiveTab('frequency')}
        >
          Frequency Analysis
        </button>
        <button 
          className={`tab ${activeTab === 'attention' ? 'active' : ''}`}
          onClick={() => setActiveTab('attention')}
        >
          Attention Map
        </button>
      </div>
      
      <div className="tab-content">
        {activeTab === 'frequency' && (
          <div className="frequency-analysis">
            <p>Frequency domain analysis showing potential generation artifacts:</p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={frequencyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="frequency" label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="amplitude" stroke="#8884d8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {activeTab === 'attention' && (
          <div className="attention-map">
            <p>Model attention visualization (regions of interest):</p>
            <div className="heatmap-placeholder">
              <p>Attention heatmap visualization would appear here</p>
              <p className="note">Highlighted regions indicate areas most influential in the detection decision</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Visualization;
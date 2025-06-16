import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { detectImage } from '../api/detction';
import { DetectionResult } from '../types';
import './Upload.css';

interface UploadProps {
  onResult: (result: DetectionResult) => void;
  onLoading: (loading: boolean) => void;
}

const Upload: React.FC<UploadProps> = ({ onResult, onLoading }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setError(null);
    
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    
    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }
    
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    
    // Upload and analyze
    onLoading(true);
    try {
      const result = await detectImage(file);
      onResult(result);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze image');
    } finally {
      onLoading(false);
    }
  }, [onResult, onLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp', '.bmp']
    },
    maxFiles: 1
  });

  return (
    <div className="upload-container">
      <div 
        {...getRootProps()} 
        className={`dropzone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <p>Click or drag another image to analyze</p>
          </div>
        ) : (
          <div className="dropzone-content">
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            <p>Drag & drop an image here, or click to select</p>
            <p className="file-types">Supports: JPEG, PNG, WebP, BMP (max 10MB)</p>
          </div>
        )}
      </div>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
    </div>
  );
};

export default Upload;
import axios from 'axios';
import { DetectionResult } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export async function detectImage(file: File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await axios.post<DetectionResult>(
      `${API_BASE_URL}/api/detect`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    
    return response.data;
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to analyze image. Please try again.');
  }
}

export async function submitFeedback(
  fileHash: string,
  correctLabel: string,
  userFeedback?: string
): Promise<void> {
  await axios.post(`${API_BASE_URL}/api/feedback`, {
    file_hash: fileHash,
    correct_label: correctLabel,
    user_feedback: userFeedback,
  });
}
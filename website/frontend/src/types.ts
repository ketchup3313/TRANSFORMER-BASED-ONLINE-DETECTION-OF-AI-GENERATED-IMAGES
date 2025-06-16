export interface DetectionResult {
  success: boolean;
  prediction: {
    is_ai_generated: boolean;
    confidence: number;
    probability_ai: number;
    probability_real: number;
    processing_time: number;
  };
  image_info: {
    width: number;
    height: number;
    format: string;
    mode: string;
    file_size_mb: number;
  };
  timestamp: string;
}
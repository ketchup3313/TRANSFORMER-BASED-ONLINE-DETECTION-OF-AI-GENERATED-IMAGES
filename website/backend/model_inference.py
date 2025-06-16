import onnxruntime as ort
import numpy as np
import time
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.threshold = 0.47 
        self._load_model()
    
    def _load_model(self):
        """ONNX model"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    async def warmup(self):
        """hot"""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        _ = self.session.run([self.output_name], {self.input_name: dummy_input})
        logger.info("Model warmup completed")
    
    async def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """start"""
        try:
            start_time = time.time()
            
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            outputs = self.session.run([self.output_name], {self.input_name: image})
            logits = outputs[0]
            
            probabilities = self._softmax(logits[0])
            ai_probability = float(probabilities[1])  # 假设索引1是AI生成类
            
            is_ai_generated = ai_probability > self.threshold
            
            confidence = ai_probability if is_ai_generated else (1 - ai_probability)
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            return {
                "is_ai_generated": bool(is_ai_generated),
                "probability": ai_probability,
                "confidence": confidence,
                "inference_time": round(inference_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise
    
    def _softmax(self, x):
        """softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def is_loaded(self) -> bool:
        """check"""
        return self.session is not None
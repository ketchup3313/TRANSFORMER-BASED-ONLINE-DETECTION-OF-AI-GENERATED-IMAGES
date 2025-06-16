from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from typing import Optional
import hashlib
import redis
import json
from datetime import datetime
import logging
from model_inference import ModelInference
from utils import validate_image, preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    cache.ping()
    CACHE_ENABLED = True
except:
    logger.warning("Redis not available, running without cache")
    CACHE_ENABLED = False
    cache = None

model = ModelInference("models/best_model.onnx")

request_queue = asyncio.Queue(maxsize=100)

@app.on_event("startup")
async def startup_event():
    """启动时预热模型"""
    logger.info("Warming up model...")
    await model.warmup()
    logger.info("Model ready for inference")

@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    检测上传的图片是否为AI生成
    """
    try:
        
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        
        is_valid, error_msg, img_info = validate_image(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        file_hash = hashlib.sha256(contents).hexdigest()

        if CACHE_ENABLED:
            cached_result = cache.get(f"result:{file_hash}")
            if cached_result:
                logger.info(f"Cache hit for {file_hash}")
                return JSONResponse(json.loads(cached_result))
        
        processed_image = preprocess_image(contents)
        
        if request_queue.full():
            raise HTTPException(status_code=503, detail="Server is busy, please try again later")
        
        result = await model.predict(processed_image)
        
        response = {
            "success": True,
            "prediction": {
                "is_ai_generated": result["is_ai_generated"],
                "confidence": float(result["confidence"]),
                "probability_ai": float(result["probability"]),
                "probability_real": float(1 - result["probability"]),
                "processing_time": result["inference_time"]
            },
            "image_info": img_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if CACHE_ENABLED:
            cache.setex(
                f"result:{file_hash}",
                86400,  # 24 hours
                json.dumps(response)
            )
        
        return JSONResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    """health check"""
    return {
        "status": "healthy",
        "model_loaded": model.is_loaded(),
        "cache_enabled": CACHE_ENABLED
    }

@app.post("/api/feedback")
async def submit_feedback(
    file_hash: str,
    correct_label: str,
    user_feedback: Optional[str] = None
):
    """feedback"""
    try:
        feedback_data = {
            "file_hash": file_hash,
            "correct_label": correct_label,
            "user_feedback": user_feedback,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        
        logger.info(f"User feedback received: {feedback_data}")
        
        return {"success": True, "message": "Thank you for your feedback"}
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
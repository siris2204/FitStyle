from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
from pathlib import Path
import uuid
import cv2
import numpy as np
import base64

import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pose_estimator import PoseEstimator
from recommender import FashionRecommender

# Initialize FastAPI app
app = FastAPI(
    title="Fashion Recommendation System",
    description="AI-powered fashion recommendations based on body measurements",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize services
pose_estimator = None
recommender = None

def get_pose_estimator():
    global pose_estimator
    if pose_estimator is None:
        pose_estimator = PoseEstimator(
            prototxt_path="models/pose/body_25/pose_deploy.prototxt",
            weights_path="models/pose/body_25/pose_iter_584000.caffemodel"
        )
    return pose_estimator

def get_recommender():
    global recommender
    if recommender is None:
        recommender = FashionRecommender(
            csv_path="data/body_measurements.csv",
            image_folder="data/fashion_images"
        )
    return recommender

# Pydantic models
class MeasurementsInput(BaseModel):
    bust: float
    waist: float
    hip: float

class MeasurementsResponse(BaseModel):
    bust: Optional[float]
    waist: Optional[float]
    hip: Optional[float]
    success: bool
    message: str

class RecommendationItem(BaseModel):
    rank: int
    filename: str
    distance: float
    bust: float
    waist: float
    hip: float
    image_base64: Optional[str]

class RecommendationsResponse(BaseModel):
    success: bool
    user_measurements: dict
    recommendations: List[RecommendationItem]

class StatsResponse(BaseModel):
    total_items: int
    bust_range: List[float]
    waist_range: List[float]
    hip_range: List[float]
    bust_mean: float
    waist_mean: float
    hip_mean: float

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    return """
    <html>
        <head>
            <title>Fashion Recommendation System</title>
        </head>
        <body>
            <h1>Fashion Recommendation API</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
            <p>Visit <a href="/frontend">/frontend</a> for the web interface</p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/extract-measurements", response_model=MeasurementsResponse)
async def extract_measurements(file: UploadFile = File(...)):
    """
    Extract body measurements from an uploaded image.
    
    Args:
        file: Uploaded image file (JPG, PNG)
        
    Returns:
        Extracted measurements (bust, waist, hip)
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    file_path = UPLOAD_DIR / f"{file_id}.{file_extension}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract measurements
        estimator = get_pose_estimator()
        measurements = estimator.get_measurements_from_image(str(file_path))
        
        # Check if measurements were extracted
        if all(v is None for v in measurements.values()):
            return MeasurementsResponse(
                bust=None,
                waist=None,
                hip=None,
                success=False,
                message="Could not detect body keypoints in the image. Please try a clearer full-body photo."
            )
        
        return MeasurementsResponse(
            bust=measurements["bust"],
            waist=measurements["waist"],
            hip=measurements["hip"],
            success=True,
            message="Measurements extracted successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            os.remove(file_path)

@app.post("/api/recommend", response_model=RecommendationsResponse)
async def get_recommendations(measurements: MeasurementsInput, k: int = Query(default=5, ge=1, le=20)):
    """
    Get fashion recommendations based on body measurements.
    
    Args:
        measurements: Body measurements (bust, waist, hip)
        k: Number of recommendations to return (1-20)
        
    Returns:
        List of recommended fashion items with images
    """
    try:
        rec = get_recommender()
        recommendations = rec.get_recommendations_with_images(
            bust=measurements.bust,
            waist=measurements.waist,
            hip=measurements.hip,
            k=k
        )
        
        return RecommendationsResponse(
            success=True,
            user_measurements={
                "bust": measurements.bust,
                "waist": measurements.waist,
                "hip": measurements.hip
            },
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend-from-image")
async def recommend_from_image(file: UploadFile = File(...), k: int = Query(default=5, ge=1, le=20)):
    """
    Upload an image and get fashion recommendations in one step.
    
    Args:
        file: Uploaded image file
        k: Number of recommendations
        
    Returns:
        Extracted measurements and recommendations
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    file_path = UPLOAD_DIR / f"{file_id}.{file_extension}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract measurements
        estimator = get_pose_estimator()
        measurements = estimator.get_measurements_from_image(str(file_path))
        
        # Check if measurements were extracted
        if all(v is None for v in measurements.values()):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Could not detect body keypoints. Please try a clearer full-body photo.",
                    "measurements": measurements,
                    "recommendations": []
                }
            )
        
        # Handle partial measurements
        bust = measurements["bust"] if measurements["bust"] else 0
        waist = measurements["waist"] if measurements["waist"] else 0
        hip = measurements["hip"] if measurements["hip"] else 0
        
        if bust == 0 and waist == 0 and hip == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No valid measurements extracted",
                    "measurements": measurements,
                    "recommendations": []
                }
            )
        
        # Get recommendations
        rec = get_recommender()
        recommendations = rec.get_recommendations_with_images(
            bust=bust,
            waist=waist,
            hip=hip,
            k=k
        )
        
        # Get annotated image
        annotated_img = estimator.draw_keypoints(str(file_path))
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "message": "Recommendations generated successfully",
            "measurements": measurements,
            "annotated_image": annotated_base64,
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up
        if file_path.exists():
            os.remove(file_path)

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the fashion dataset."""
    try:
        rec = get_recommender()
        return rec.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Get a specific fashion image by filename."""
    try:
        rec = get_recommender()
        image_base64 = rec.get_image_base64(filename)
        
        if image_base64 is None:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {"filename": filename, "image_base64": image_base64}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
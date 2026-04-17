import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import base64

class FashionRecommender:
    """KNN-based fashion recommendation system."""
    
    def __init__(self, 
                 csv_path: str = "data/body_measurements.csv",
                 image_folder: str = "data/fashion_images",
                 n_neighbors: int = 5):
        """
        Initialize the recommender.
        
        Args:
            csv_path: Path to the measurements CSV file
            image_folder: Path to the fashion images folder
            n_neighbors: Number of recommendations to return
        """
        self.csv_path = Path(csv_path)
        self.image_folder = Path(image_folder)
        self.n_neighbors = n_neighbors
        self.knn = None
        self.df = None
        self.features = None
        self.filenames = None
        
        self._load_data()
    
    def _load_data(self):
        """Load the measurements data and fit the KNN model."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Clean data - drop rows with missing values
        self.df = self.df.dropna(subset=["bust", "waist", "hip"])
        
        self.features = self.df[["bust", "waist", "hip"]].values
        self.filenames = self.df["filename"].values
        
        # Fit KNN model
        self.knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self.features)))
        self.knn.fit(self.features)
    
    def recommend(self, 
                  bust: float, 
                  waist: float, 
                  hip: float, 
                  k: int = None) -> List[Dict]:
        """
        Get fashion recommendations based on body measurements.
        
        Args:
            bust: Bust measurement
            waist: Waist measurement
            hip: Hip measurement
            k: Number of recommendations (defaults to n_neighbors)
            
        Returns:
            List of recommendation dictionaries with filename, distance, and measurements
        """
        if k is None:
            k = self.n_neighbors
        
        k = min(k, len(self.features))
        
        input_vector = np.array([[bust, waist, hip]])
        distances, indices = self.knn.kneighbors(input_vector, n_neighbors=k)
        
        recommendations = []
        for i, idx in enumerate(indices[0]):
            rec = {
                "rank": i + 1,
                "filename": self.filenames[idx],
                "distance": float(distances[0][i]),
                "bust": float(self.features[idx][0]),
                "waist": float(self.features[idx][1]),
                "hip": float(self.features[idx][2])
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_image_base64(self, filename: str) -> Optional[str]:
        """
        Get base64 encoded image.
        
        Args:
            filename: Name of the image file
            
        Returns:
            Base64 encoded image string or None if not found
        """
        image_path = self.image_folder / filename
        if not image_path.exists():
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Resize for faster transfer
        max_size = 400
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
    
    def get_recommendations_with_images(self, 
                                         bust: float, 
                                         waist: float, 
                                         hip: float, 
                                         k: int = None) -> List[Dict]:
        """
        Get recommendations with base64 encoded images included.
        
        Args:
            bust: Bust measurement
            waist: Waist measurement
            hip: Hip measurement
            k: Number of recommendations
            
        Returns:
            List of recommendations with image data included
        """
        recommendations = self.recommend(bust, waist, hip, k)
        
        for rec in recommendations:
            rec["image_base64"] = self.get_image_base64(rec["filename"])
        
        return recommendations
    
    def add_measurements(self, filename: str, bust: float, waist: float, hip: float):
        """
        Add new measurements to the dataset.
        
        Args:
            filename: Image filename
            bust: Bust measurement
            waist: Waist measurement
            hip: Hip measurement
        """
        new_row = pd.DataFrame([{
            "filename": filename,
            "bust": bust,
            "waist": waist,
            "hip": hip
        }])
        
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False)
        
        # Refit the model
        self.features = self.df[["bust", "waist", "hip"]].values
        self.filenames = self.df["filename"].values
        self.knn.fit(self.features)
    
    def get_stats(self) -> Dict:
        """Get statistics about the dataset."""
        return {
            "total_items": len(self.df),
            "bust_range": [float(self.df["bust"].min()), float(self.df["bust"].max())],
            "waist_range": [float(self.df["waist"].min()), float(self.df["waist"].max())],
            "hip_range": [float(self.df["hip"].min()), float(self.df["hip"].max())],
            "bust_mean": float(self.df["bust"].mean()),
            "waist_mean": float(self.df["waist"].mean()),
            "hip_mean": float(self.df["hip"].mean())
        }
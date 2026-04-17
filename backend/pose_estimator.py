import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

class PoseEstimator:
    """OpenPose-based pose estimation using OpenCV DNN module."""
    
    # BODY_25 keypoint indices
    KEYPOINT_NAMES = {
        0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
        5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip",
        9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee",
        14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar",
        19: "LBigToe", 20: "LSmallToe", 21: "LHeel", 22: "RBigToe",
        23: "RSmallToe", 24: "RHeel"
    }
    
    def __init__(self, 
                 prototxt_path: str = "models/pose/body_25/pose_deploy.prototxt",
                 weights_path: str = "models/pose/body_25/pose_iter_584000.caffemodel",
                 threshold: float = 0.1):
        """
        Initialize the pose estimator.
        
        Args:
            prototxt_path: Path to the prototxt file
            weights_path: Path to the caffemodel weights file
            threshold: Confidence threshold for keypoint detection
        """
        self.prototxt_path = Path(prototxt_path)
        self.weights_path = Path(weights_path)
        self.threshold = threshold
        self.net = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the OpenPose model."""
        if not self.prototxt_path.exists():
            raise FileNotFoundError(f"Prototxt file not found: {self.prototxt_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        self.net = cv2.dnn.readNetFromCaffe(
            str(self.prototxt_path), 
            str(self.weights_path)
        )
        
        # Use CPU only (no GPU)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def extract_keypoints(self, image_path: str) -> Optional[List[Optional[Tuple[int, int]]]]:
        """
        Extract keypoints from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of 25 keypoints (x, y) or None if point not detected
        """
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        
        return self._process_frame(frame)
    
    def extract_keypoints_from_array(self, image_array: np.ndarray) -> Optional[List[Optional[Tuple[int, int]]]]:
        """
        Extract keypoints from a numpy array.
        
        Args:
            image_array: Image as numpy array (BGR format)
            
        Returns:
            List of 25 keypoints (x, y) or None if point not detected
        """
        return self._process_frame(image_array)
    
    def _process_frame(self, frame: np.ndarray) -> List[Optional[Tuple[int, int]]]:
        """Process a single frame and extract keypoints."""
        frame_height, frame_width = frame.shape[:2]
        
        # Create blob from image
        inp_blob = cv2.dnn.blobFromImage(
            frame, 
            1.0 / 255, 
            (368, 368),
            (0, 0, 0), 
            swapRB=False, 
            crop=False
        )
        
        self.net.setInput(inp_blob)
        output = self.net.forward()
        
        H, W = output.shape[2], output.shape[3]
        
        points = []
        for i in range(25):  # BODY_25 has 25 keypoints
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            
            x = int((frame_width * point[0]) / W)
            y = int((frame_height * point[1]) / H)
            
            if prob > self.threshold:
                points.append((x, y))
            else:
                points.append(None)
        
        return points
    
    def calculate_measurements(self, keypoints: List[Optional[Tuple[int, int]]]) -> Dict[str, Optional[float]]:
        """
        Calculate body measurements from keypoints.
        
        Args:
            keypoints: List of detected keypoints
            
        Returns:
            Dictionary with bust, waist, and hip measurements
        """
        def euclidean(p1, p2):
            if p1 is None or p2 is None:
                return None
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Bust: RShoulder (2) to LShoulder (5)
        bust = euclidean(keypoints[2], keypoints[5])
        
        # Waist: MidHip area - using RHip (9) to LHip (12) as proxy
        # Note: BODY_25 index 8 is MidHip, 9 is RHip, 12 is LHip
        waist = euclidean(keypoints[8], keypoints[11]) if len(keypoints) > 11 else None
        
        # Hip: RHip (9) to LHip (12)
        hip = euclidean(keypoints[9], keypoints[12])
        
        return {
            "bust": bust,
            "waist": waist,
            "hip": hip
        }
    
    def get_measurements_from_image(self, image_path: str) -> Dict[str, Optional[float]]:
        """
        Convenience method to get measurements directly from image path.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with bust, waist, and hip measurements
        """
        keypoints = self.extract_keypoints(image_path)
        if keypoints is None:
            return {"bust": None, "waist": None, "hip": None}
        return self.calculate_measurements(keypoints)
    
    def draw_keypoints(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Draw detected keypoints on the image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save the annotated image
            
        Returns:
            Annotated image as numpy array
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        keypoints = self._process_frame(frame)
        
        # Draw keypoints
        for i, point in enumerate(keypoints):
            if point is not None:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (point[0] + 5, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Draw connections for body measurements
        connections = [
            (2, 5, (255, 0, 0)),   # Bust (shoulders)
            (8, 11, (0, 255, 0)),  # Waist approximation
            (9, 12, (0, 0, 255))   # Hip
        ]
        
        for start, end, color in connections:
            if keypoints[start] is not None and keypoints[end] is not None:
                cv2.line(frame, keypoints[start], keypoints[end], color, 2)
        
        if output_path:
            cv2.imwrite(output_path, frame)
        
        return frame
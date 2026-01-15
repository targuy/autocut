"""Face detection using YuNet ONNX model with multi-provider support."""

import csv
from pathlib import Path
from typing import List, Tuple, Optional
import math

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class FaceDetector:
    """Face detector using YuNet ONNX model."""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """Initialize face detector.
        
        Args:
            model_path: Path to YuNet ONNX model
            providers: List of ONNX Runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        Raises:
            ImportError: If required libraries are not installed
        """
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")
        
        if not ORT_AVAILABLE:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Model expects 160x120 input
        self.input_size = (160, 120)
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.6) -> List[dict]:
        """Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold (default 0.6)
        
        Returns:
            List of detection dicts with keys: bbox, landmarks, confidence, yaw, pitch, roll
        """
        if image is None or image.size == 0:
            return []
        
        # Preprocess
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        
        # Parse detections
        detections = []
        if outputs and len(outputs) > 0:
            # YuNet output format: [x, y, w, h, landmarks (10 values), conf]
            for detection in outputs[0][0]:
                conf = detection[14]
                if conf < conf_threshold:
                    continue
                
                # Scale bbox to original image size
                x, y, box_w, box_h = detection[0:4]
                x = int(x * w / self.input_size[0])
                y = int(y * h / self.input_size[1])
                box_w = int(box_w * w / self.input_size[0])
                box_h = int(box_h * h / self.input_size[1])
                
                # Scale landmarks
                landmarks = []
                for i in range(5):
                    lx = int(detection[4 + i * 2] * w / self.input_size[0])
                    ly = int(detection[5 + i * 2] * h / self.input_size[1])
                    landmarks.append((lx, ly))
                
                # Estimate head pose (simplified)
                yaw, pitch, roll = self._estimate_head_pose(landmarks)
                
                detections.append({
                    'bbox': (x, y, box_w, box_h),
                    'landmarks': landmarks,
                    'confidence': float(conf),
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                })
        
        return detections
    
    def _estimate_head_pose(self, landmarks: List[Tuple[int, int]]) -> Tuple[float, float, float]:
        """Estimate head pose from landmarks (simplified).
        
        Args:
            landmarks: List of 5 facial landmarks (eyes, nose, mouth corners)
        
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        if len(landmarks) < 5:
            return (0.0, 0.0, 0.0)
        
        # Simplified pose estimation using eye and nose positions
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
        
        # Calculate roll (head tilt) from eye line
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll = math.degrees(math.atan2(dy, dx))
        
        # Simplified yaw estimation (left/right turn)
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_x = nose[0]
        yaw = (nose_x - eye_center_x) * 0.5  # Rough approximation
        
        # Simplified pitch estimation (up/down tilt)
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        nose_y = nose[1]
        pitch = (nose_y - eye_center_y) * 0.3  # Rough approximation
        
        return (yaw, pitch, roll)


def detect_faces_in_video(
    video_path: str,
    model_path: str,
    output_csv: str,
    providers: List[str] = None,
    skip_frames: int = 30,
    draw_output_dir: Optional[str] = None,
):
    """Detect faces in video frames and save results to CSV.
    
    Args:
        video_path: Path to video file
        model_path: Path to YuNet ONNX model
        output_csv: Output CSV file path
        providers: ONNX Runtime providers
        skip_frames: Process every Nth frame (default 30)
        draw_output_dir: Optional directory to save frames with drawn detections
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required")
    
    detector = FaceDetector(model_path, providers)
    cap = cv2.VideoCapture(video_path)
    
    results = []
    frame_idx = 0
    
    if draw_output_dir:
        Path(draw_output_dir).mkdir(parents=True, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames == 0:
            detections = detector.detect(frame)
            
            for det in detections:
                x, y, w, h = det['bbox']
                results.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': det['confidence'],
                    'yaw': det['yaw'],
                    'pitch': det['pitch'],
                    'roll': det['roll'],
                })
            
            # Optionally draw detections
            if draw_output_dir and detections:
                for det in detections:
                    x, y, w, h = det['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    for lx, ly in det['landmarks']:
                        cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
                
                output_path = Path(draw_output_dir) / f'frame_{frame_idx:06d}.jpg'
                cv2.imwrite(str(output_path), frame)
        
        frame_idx += 1
    
    cap.release()
    
    # Save results to CSV
    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


def detect_faces_in_frames(
    frames_dir: str,
    model_path: str,
    output_csv: str,
    providers: List[str] = None,
    draw_output_dir: Optional[str] = None,
):
    """Detect faces in a directory of frames.
    
    Args:
        frames_dir: Directory containing frame images
        model_path: Path to YuNet ONNX model
        output_csv: Output CSV file path
        providers: ONNX Runtime providers
        draw_output_dir: Optional directory to save frames with drawn detections
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required")
    
    detector = FaceDetector(model_path, providers)
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob('*.jpg')) + sorted(frames_path.glob('*.png'))
    
    results = []
    
    if draw_output_dir:
        Path(draw_output_dir).mkdir(parents=True, exist_ok=True)
    
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
        
        detections = detector.detect(frame)
        
        for det in detections:
            x, y, w, h = det['bbox']
            results.append({
                'frame_file': frame_file.name,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'confidence': det['confidence'],
                'yaw': det['yaw'],
                'pitch': det['pitch'],
                'roll': det['roll'],
            })
        
        # Optionally draw detections
        if draw_output_dir and detections:
            for det in detections:
                x, y, w, h = det['bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                for lx, ly in det['landmarks']:
                    cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
            
            output_path = Path(draw_output_dir) / frame_file.name
            cv2.imwrite(str(output_path), frame)
    
    # Save results to CSV
    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

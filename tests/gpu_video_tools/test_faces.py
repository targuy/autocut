"""Tests for face detection module."""

import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from gpu_video_tools.faces import (
    FaceDetector,
    detect_faces_in_video,
    detect_faces_in_frames,
    CV2_AVAILABLE,
    ORT_AVAILABLE,
)


def test_imports_available():
    """Test that import availability is properly tracked."""
    assert isinstance(CV2_AVAILABLE, bool)
    assert isinstance(ORT_AVAILABLE, bool)


def test_face_detector_import_error_cv2():
    """Test FaceDetector raises ImportError when cv2 not available."""
    with patch('gpu_video_tools.faces.CV2_AVAILABLE', False):
        with pytest.raises(ImportError, match="opencv-python is required"):
            FaceDetector('model.onnx')


def test_face_detector_import_error_ort():
    """Test FaceDetector raises ImportError when onnxruntime not available."""
    with patch('gpu_video_tools.faces.CV2_AVAILABLE', True):
        with patch('gpu_video_tools.faces.ORT_AVAILABLE', False):
            with pytest.raises(ImportError, match="onnxruntime is required"):
                FaceDetector('model.onnx')


@pytest.mark.skipif(not (CV2_AVAILABLE and ORT_AVAILABLE), reason="cv2 or onnxruntime not installed")
def test_face_detector_initialization_real():
    """Test FaceDetector initialization (requires real libraries)."""
    # This would require actual ONNX model file
    # Skipped in unit tests, covered in integration tests
    pass


def test_estimate_head_pose():
    """Test head pose estimation from landmarks (internal method)."""
    mock_ort = MagicMock()
    with patch('gpu_video_tools.faces.CV2_AVAILABLE', True):
        with patch('gpu_video_tools.faces.ORT_AVAILABLE', True):
            with patch('gpu_video_tools.faces.ort', mock_ort, create=True):
                detector = FaceDetector('model.onnx')
                
                # Create mock landmarks (5 points: left eye, right eye, nose, left mouth, right mouth)
                landmarks = [
                    (100, 100),  # left eye
                    (200, 100),  # right eye
                    (150, 150),  # nose
                    (120, 200),  # left mouth
                    (180, 200),  # right mouth
                ]
                
                yaw, pitch, roll = detector._estimate_head_pose(landmarks)
                
                # Should return some reasonable values
                assert isinstance(yaw, float)
                assert isinstance(pitch, float)
                assert isinstance(roll, float)
                
                # Angles should be in reasonable range
                assert -180 <= yaw <= 180
                assert -180 <= pitch <= 180
                assert -180 <= roll <= 180


def test_estimate_head_pose_frontal():
    """Test head pose for frontal face."""
    mock_ort = MagicMock()
    with patch('gpu_video_tools.faces.CV2_AVAILABLE', True):
        with patch('gpu_video_tools.faces.ORT_AVAILABLE', True):
            with patch('gpu_video_tools.faces.ort', mock_ort, create=True):
                detector = FaceDetector('model.onnx')
                
                # Symmetric landmarks for frontal face
                landmarks = [
                    (100, 100),  # left eye
                    (200, 100),  # right eye
                    (150, 150),  # nose (centered)
                    (125, 200),  # left mouth
                    (175, 200),  # right mouth
                ]
                
                yaw, pitch, roll = detector._estimate_head_pose(landmarks)
                
                # For frontal face, roll should be close to 0
                assert abs(roll) < 20


@patch('gpu_video_tools.faces.ORT_AVAILABLE', True)
@patch('gpu_video_tools.faces.CV2_AVAILABLE', True)
def test_face_detector_detect_mocked():
    """Test FaceDetector.detect with mocked dependencies."""
    import numpy as np
    
    # Mock onnxruntime module  
    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    # Mock cv2 module
    mock_cv2 = MagicMock()
    
    with patch('gpu_video_tools.faces.ort', mock_ort, create=True):
        with patch('gpu_video_tools.faces.cv2', mock_cv2, create=True):
            # Mock input
            mock_input = MagicMock()
            mock_input.name = 'input'
            mock_session.get_inputs.return_value = [mock_input]
            
            # Mock detection output
            # YuNet format: [x, y, w, h, 10 landmark coords, confidence]
            # Total: 4 + 10 + 1 = 15 elements
            mock_detection = [10, 10, 100, 100] + [0]*10 + [0.95]
            mock_session.run.return_value = [[[mock_detection]]]
            
            # Mock cv2.dnn.blobFromImage
            mock_cv2.dnn.blobFromImage.return_value = MagicMock()
            
            # Create detector
            detector = FaceDetector('model.onnx')
            
            # Mock image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Run detection
            detections = detector.detect(image, conf_threshold=0.5)
            
            # Should get detections
            assert len(detections) > 0
            assert 'bbox' in detections[0]
            assert 'confidence' in detections[0]


def test_detect_faces_in_video_not_implemented():
    """Test detect_faces_in_video function exists."""
    # Function should exist and be callable
    assert callable(detect_faces_in_video)


def test_detect_faces_in_frames_not_implemented():
    """Test detect_faces_in_frames function exists."""
    # Function should exist and be callable
    assert callable(detect_faces_in_frames)


@patch('gpu_video_tools.faces.CV2_AVAILABLE', True)
def test_face_detector_empty_image():
    """Test detecting faces in empty/None image."""
    import numpy as np
    
    mock_ort = MagicMock()
    with patch('gpu_video_tools.faces.ort', mock_ort, create=True):
        with patch('gpu_video_tools.faces.ORT_AVAILABLE', True):
            detector = FaceDetector('model.onnx')
            # Test with None
            detections = detector.detect(None)
            assert detections == []
            
            # Test with empty array
            empty = np.array([])
            detections = detector.detect(empty)
            assert detections == []


def test_face_detector_providers():
    """Test FaceDetector with custom providers."""
    mock_ort = MagicMock()
    with patch('gpu_video_tools.faces.ort', mock_ort, create=True):
        with patch('gpu_video_tools.faces.CV2_AVAILABLE', True):
            with patch('gpu_video_tools.faces.ORT_AVAILABLE', True):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                detector = FaceDetector('model.onnx', providers=providers)
                
                # Verify providers were passed to session
                mock_ort.InferenceSession.assert_called_once()
                call_kwargs = mock_ort.InferenceSession.call_args[1]
                assert call_kwargs['providers'] == providers


def test_face_detector_default_providers():
    """Test FaceDetector with default providers."""
    mock_ort = MagicMock()
    with patch('gpu_video_tools.faces.ort', mock_ort, create=True):
        with patch('gpu_video_tools.faces.CV2_AVAILABLE', True):
            with patch('gpu_video_tools.faces.ORT_AVAILABLE', True):
                detector = FaceDetector('model.onnx')
                
                # Should default to CPU
                call_kwargs = mock_ort.InferenceSession.call_args[1]
                assert call_kwargs['providers'] == ['CPUExecutionProvider']

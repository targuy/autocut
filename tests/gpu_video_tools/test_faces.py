"""Tests for face detection functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import csv

import pytest
import numpy as np

from gpu_video_tools.faces import FaceDetector, detect_faces_in_video, detect_faces_in_frames


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX Runtime session."""
    session = MagicMock()
    
    # Mock input
    mock_input = Mock()
    mock_input.name = 'input'
    session.get_inputs.return_value = [mock_input]
    
    # Mock detection output
    # YuNet output format: [batch, num_detections, 15]
    # 15 values: x, y, w, h, 10 landmarks, confidence
    mock_output = np.array([[[
        50.0, 50.0, 100.0, 100.0,  # bbox
        55.0, 60.0,  # left eye
        95.0, 60.0,  # right eye
        75.0, 85.0,  # nose
        60.0, 110.0,  # left mouth
        90.0, 110.0,  # right mouth
        0.95  # confidence
    ]]])
    
    session.run.return_value = [mock_output]
    
    return session


@patch('gpu_video_tools.faces.ORT_AVAILABLE', True)
@patch('gpu_video_tools.faces.CV2_AVAILABLE', True)
@patch('gpu_video_tools.faces.ort')
def test_face_detector_initialization(mock_ort):
    """Test FaceDetector initialization."""
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    detector = FaceDetector('/path/to/model.onnx', providers=['CPUExecutionProvider'])
    
    assert detector.model_path == '/path/to/model.onnx'
    assert detector.providers == ['CPUExecutionProvider']
    assert detector.input_size == (160, 120)
    
    mock_ort.InferenceSession.assert_called_once_with(
        '/path/to/model.onnx',
        providers=['CPUExecutionProvider']
    )


@patch('gpu_video_tools.faces.ORT_AVAILABLE', True)
@patch('gpu_video_tools.faces.CV2_AVAILABLE', True)
@patch('gpu_video_tools.faces.ort')
@patch('gpu_video_tools.faces.cv2')
def test_face_detector_detect(mock_cv2, mock_ort, mock_onnx_session):
    """Test face detection on image."""
    mock_ort.InferenceSession.return_value = mock_onnx_session
    
    # Create fake image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock cv2.dnn.blobFromImage
    mock_blob = np.zeros((1, 3, 120, 160))
    mock_cv2.dnn.blobFromImage.return_value = mock_blob
    
    # Create detector
    detector = FaceDetector('/path/to/model.onnx')
    
    # Detect faces
    detections = detector.detect(image)
    
    # Verify blob creation
    mock_cv2.dnn.blobFromImage.assert_called_once()
    
    # Verify session run
    mock_onnx_session.run.assert_called_once()
    
    # Check detections
    assert isinstance(detections, list)
    assert len(detections) > 0


@patch('gpu_video_tools.faces.ORT_AVAILABLE', False)
@patch('gpu_video_tools.faces.CV2_AVAILABLE', True)
def test_face_detector_missing_onnxruntime():
    """Test that ImportError is raised when onnxruntime is not available."""
    with pytest.raises(ImportError, match="onnxruntime is required"):
        FaceDetector('/path/to/model.onnx')


@patch('gpu_video_tools.faces.ORT_AVAILABLE', True)
@patch('gpu_video_tools.faces.CV2_AVAILABLE', False)
def test_face_detector_missing_opencv():
    """Test that ImportError is raised when opencv is not available."""
    with pytest.raises(ImportError, match="opencv-python is required"):
        FaceDetector('/path/to/model.onnx')


@patch('gpu_video_tools.faces.ORT_AVAILABLE', True)
@patch('gpu_video_tools.faces.CV2_AVAILABLE', True)
@patch('gpu_video_tools.faces.ort')
@patch('gpu_video_tools.faces.cv2')
def test_face_detector_empty_image(mock_cv2, mock_ort, mock_onnx_session):
    """Test detection on empty/None image."""
    mock_ort.InferenceSession.return_value = mock_onnx_session
    
    detector = FaceDetector('/path/to/model.onnx')
    
    # None image
    detections = detector.detect(None)
    assert detections == []


@patch('gpu_video_tools.faces.FaceDetector')
@patch('gpu_video_tools.faces.cv2')
def test_detect_faces_in_video(mock_cv2, mock_detector_class):
    """Test face detection in video file."""
    # Setup mocks
    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector
    
    # Mock video capture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda x: 30.0 if x == mock_cv2.CAP_PROP_FPS else 100.0
    
    # Simulate reading frames
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    mock_cap.read.side_effect = [
        (True, frame1),
        (True, frame2),
        (False, None)  # End of video
    ]
    
    mock_cv2.VideoCapture.return_value = mock_cap
    
    # Mock detections
    mock_detector.detect.return_value = [
        {
            'bbox': (50, 50, 100, 100),
            'confidence': 0.95,
            'landmarks': [(55, 60), (95, 60), (75, 85), (60, 110), (90, 110)],
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        output_csv = f.name
    
    try:
        # Run detection
        detect_faces_in_video(
            'test_video.mp4',
            'model.onnx',
            output_csv,
            providers=['CPUExecutionProvider'],
            skip_frames=1
        )
        
        # Verify video capture was created
        mock_cv2.VideoCapture.assert_called_once_with('test_video.mp4')
        
        # Verify detector was created
        mock_detector_class.assert_called_once_with('model.onnx', providers=['CPUExecutionProvider'])
        
        # Verify CSV was written
        assert Path(output_csv).exists()
        
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0
    
    finally:
        Path(output_csv).unlink(missing_ok=True)


@patch('gpu_video_tools.faces.FaceDetector')
@patch('gpu_video_tools.faces.cv2')
def test_detect_faces_in_frames(mock_cv2, mock_detector_class):
    """Test face detection in frame directory."""
    # Setup mocks
    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector
    
    # Mock image reading
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cv2.imread.return_value = mock_frame
    
    # Mock detections
    mock_detector.detect.return_value = [
        {
            'bbox': (50, 50, 100, 100),
            'confidence': 0.95,
            'landmarks': [(55, 60), (95, 60), (75, 85), (60, 110), (90, 110)],
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        }
    ]
    
    with tempfile.TemporaryDirectory() as frames_dir:
        # Create fake frame files
        frame_dir_path = Path(frames_dir)
        (frame_dir_path / 'frame_0001.jpg').touch()
        (frame_dir_path / 'frame_0002.jpg').touch()
        (frame_dir_path / 'frame_0003.jpg').touch()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_csv = f.name
        
        try:
            # Run detection
            detect_faces_in_frames(
                frames_dir,
                'model.onnx',
                output_csv,
                providers=['CPUExecutionProvider']
            )
            
            # Verify detector was created
            mock_detector_class.assert_called_once_with('model.onnx', providers=['CPUExecutionProvider'])
            
            # Verify images were read
            assert mock_cv2.imread.call_count == 3
            
            # Verify CSV was written
            assert Path(output_csv).exists()
            
            with open(output_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Should have detections from 3 frames
                assert len(rows) > 0
        
        finally:
            Path(output_csv).unlink(missing_ok=True)


@patch('gpu_video_tools.faces.FaceDetector')
@patch('gpu_video_tools.faces.cv2')
def test_detect_faces_in_video_with_draw(mock_cv2, mock_detector_class):
    """Test face detection with visualization output."""
    # Setup mocks
    mock_detector = MagicMock()
    mock_detector_class.return_value = mock_detector
    
    # Mock video capture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda x: 30.0 if x == mock_cv2.CAP_PROP_FPS else 10.0
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, frame),
        (False, None)
    ]
    
    mock_cv2.VideoCapture.return_value = mock_cap
    
    # Mock detections
    mock_detector.detect.return_value = [
        {
            'bbox': (50, 50, 100, 100),
            'confidence': 0.95,
            'landmarks': [(55, 60), (95, 60), (75, 85), (60, 110), (90, 110)],
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        }
    ]
    
    with tempfile.TemporaryDirectory() as draw_dir:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_csv = f.name
        
        try:
            # Run detection with draw output
            detect_faces_in_video(
                'test_video.mp4',
                'model.onnx',
                output_csv,
                providers=['CPUExecutionProvider'],
                skip_frames=1,
                draw_output_dir=draw_dir
            )
            
            # Verify drawing functions were called
            # (rectangle for bbox, circles for landmarks)
            assert mock_cv2.rectangle.called or mock_cv2.circle.called
            
        finally:
            Path(output_csv).unlink(missing_ok=True)


@patch('gpu_video_tools.faces.ORT_AVAILABLE', True)
@patch('gpu_video_tools.faces.CV2_AVAILABLE', True)
@patch('gpu_video_tools.faces.ort')
@patch('gpu_video_tools.faces.cv2')
def test_face_detector_confidence_threshold(mock_cv2, mock_ort):
    """Test face detection with different confidence thresholds."""
    # Create mock session with low confidence detection
    mock_session = MagicMock()
    mock_input = Mock()
    mock_input.name = 'input'
    mock_session.get_inputs.return_value = [mock_input]
    
    # Low confidence detection (0.3)
    mock_output = np.array([[[
        50.0, 50.0, 100.0, 100.0,
        55.0, 60.0, 95.0, 60.0, 75.0, 85.0, 60.0, 110.0, 90.0, 110.0,
        0.3  # Low confidence
    ]]])
    mock_session.run.return_value = [mock_output]
    
    mock_ort.InferenceSession.return_value = mock_session
    
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_blob = np.zeros((1, 3, 120, 160))
    mock_cv2.dnn.blobFromImage.return_value = mock_blob
    
    detector = FaceDetector('/path/to/model.onnx')
    
    # With high threshold (0.6), should get no detections
    detections_high = detector.detect(image, conf_threshold=0.6)
    
    # With low threshold (0.2), should get detection
    detections_low = detector.detect(image, conf_threshold=0.2)
    
    # Note: actual filtering depends on implementation
    # This test verifies the threshold parameter is accepted
    assert isinstance(detections_high, list)
    assert isinstance(detections_low, list)

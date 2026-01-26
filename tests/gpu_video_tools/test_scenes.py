"""Tests for scene detection functionality."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from gpu_video_tools.scenes import detect_scenes, save_scenes_csv


@pytest.fixture
def mock_scene_list():
    """Mock scene list with timecode objects."""
    # Create mock timecode objects
    start1 = Mock()
    start1.get_seconds.return_value = 0.0
    end1 = Mock()
    end1.get_seconds.return_value = 5.5
    
    start2 = Mock()
    start2.get_seconds.return_value = 5.5
    end2 = Mock()
    end2.get_seconds.return_value = 12.3
    
    start3 = Mock()
    start3.get_seconds.return_value = 12.3
    end3 = Mock()
    end3.get_seconds.return_value = 20.0
    
    return [(start1, end1), (start2, end2), (start3, end3)]


@patch('gpu_video_tools.scenes.SCENEDETECT_AVAILABLE', True)
@patch('gpu_video_tools.scenes.VideoManager')
@patch('gpu_video_tools.scenes.SceneManager')
def test_detect_scenes_basic(mock_scene_manager_class, mock_video_manager_class, mock_scene_list):
    """Test basic scene detection."""
    # Setup mocks
    mock_vm = MagicMock()
    mock_video_manager_class.return_value = mock_vm
    
    mock_sm = MagicMock()
    mock_sm.get_scene_list.return_value = mock_scene_list
    mock_scene_manager_class.return_value = mock_sm
    
    # Mock framerate
    mock_timecode = Mock()
    mock_timecode.framerate = 30
    mock_vm.get_base_timecode.return_value = mock_timecode
    
    # Run detection
    scenes = detect_scenes('test_video.mp4')
    
    # Verify results
    assert len(scenes) == 3
    assert scenes[0] == (0.0, 5.5)
    assert scenes[1] == (5.5, 12.3)
    assert scenes[2] == (12.3, 20.0)
    
    # Verify VideoManager was called correctly
    mock_video_manager_class.assert_called_once_with(['test_video.mp4'])
    mock_vm.start.assert_called_once()
    mock_vm.release.assert_called_once()
    
    # Verify SceneManager was set up
    mock_sm.add_detector.assert_called_once()
    mock_sm.detect_scenes.assert_called_once()


@patch('gpu_video_tools.scenes.SCENEDETECT_AVAILABLE', True)
@patch('gpu_video_tools.scenes.VideoManager')
@patch('gpu_video_tools.scenes.SceneManager')
def test_detect_scenes_custom_threshold(mock_scene_manager_class, mock_video_manager_class, mock_scene_list):
    """Test scene detection with custom threshold."""
    # Setup mocks
    mock_vm = MagicMock()
    mock_video_manager_class.return_value = mock_vm
    
    mock_sm = MagicMock()
    mock_sm.get_scene_list.return_value = mock_scene_list
    mock_scene_manager_class.return_value = mock_sm
    
    mock_timecode = Mock()
    mock_timecode.framerate = 30
    mock_vm.get_base_timecode.return_value = mock_timecode
    
    # Run with custom threshold
    scenes = detect_scenes('test_video.mp4', threshold=35.0, min_scene_len=2.0)
    
    assert len(scenes) == 3
    mock_sm.add_detector.assert_called_once()


@patch('gpu_video_tools.scenes.SCENEDETECT_AVAILABLE', True)
@patch('gpu_video_tools.scenes.VideoManager')
@patch('gpu_video_tools.scenes.SceneManager')
def test_detect_scenes_empty_result(mock_scene_manager_class, mock_video_manager_class):
    """Test scene detection with no scenes detected."""
    # Setup mocks
    mock_vm = MagicMock()
    mock_video_manager_class.return_value = mock_vm
    
    mock_sm = MagicMock()
    mock_sm.get_scene_list.return_value = []
    mock_scene_manager_class.return_value = mock_sm
    
    mock_timecode = Mock()
    mock_timecode.framerate = 30
    mock_vm.get_base_timecode.return_value = mock_timecode
    
    # Run detection
    scenes = detect_scenes('test_video.mp4')
    
    # Should return empty list
    assert scenes == []


@patch('gpu_video_tools.scenes.SCENEDETECT_AVAILABLE', False)
def test_detect_scenes_missing_dependency():
    """Test that ImportError is raised when scenedetect is not available."""
    with pytest.raises(ImportError, match="scenedetect is required"):
        detect_scenes('test_video.mp4')


def test_save_scenes_csv():
    """Test saving scenes to CSV."""
    scenes = [
        (0.0, 5.5),
        (5.5, 12.3),
        (12.3, 20.0),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        output_path = f.name
    
    try:
        # Save to CSV
        save_scenes_csv(scenes, output_path)
        
        # Read and verify
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Check header
        assert rows[0] == ['scene_number', 'start_time', 'end_time', 'duration']
        
        # Check data rows
        assert len(rows) == 4  # Header + 3 scenes
        
        assert rows[1][0] == '1'
        assert rows[1][1] == '0.000'
        assert rows[1][2] == '5.500'
        assert rows[1][3] == '5.500'
        
        assert rows[2][0] == '2'
        assert rows[2][1] == '5.500'
        assert rows[2][2] == '12.300'
        assert float(rows[2][3]) == pytest.approx(6.8, abs=0.01)
        
        assert rows[3][0] == '3'
    
    finally:
        # Cleanup
        Path(output_path).unlink(missing_ok=True)


def test_save_scenes_csv_empty():
    """Test saving empty scene list to CSV."""
    scenes = []
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        output_path = f.name
    
    try:
        # Save to CSV
        save_scenes_csv(scenes, output_path)
        
        # Read and verify
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Should only have header
        assert len(rows) == 1
        assert rows[0] == ['scene_number', 'start_time', 'end_time', 'duration']
    
    finally:
        # Cleanup
        Path(output_path).unlink(missing_ok=True)


def test_save_scenes_csv_single_scene():
    """Test saving single scene to CSV."""
    scenes = [(0.0, 10.5)]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        output_path = f.name
    
    try:
        save_scenes_csv(scenes, output_path)
        
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        assert len(rows) == 2  # Header + 1 scene
        assert rows[1][0] == '1'
        assert rows[1][1] == '0.000'
        assert rows[1][2] == '10.500'
        assert rows[1][3] == '10.500'
    
    finally:
        Path(output_path).unlink(missing_ok=True)

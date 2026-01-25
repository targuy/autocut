"""Tests for scene detection module."""

import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from gpu_video_tools.scenes import (
    detect_scenes,
    save_scenes_csv,
    SCENEDETECT_AVAILABLE,
)


def test_scenedetect_import():
    """Test that scenedetect availability is properly tracked."""
    # Just verify the constant exists
    assert isinstance(SCENEDETECT_AVAILABLE, bool)


@pytest.mark.skipif(not SCENEDETECT_AVAILABLE, reason="scenedetect not installed")
def test_detect_scenes_not_implemented():
    """Test detect_scenes with actual library (if available)."""
    # This test would require a real video file, so we skip it for unit tests
    # Integration tests should cover this
    pass


def test_detect_scenes_import_error():
    """Test detect_scenes raises ImportError when scenedetect not available."""
    with patch('gpu_video_tools.scenes.SCENEDETECT_AVAILABLE', False):
        with pytest.raises(ImportError, match="scenedetect is required"):
            detect_scenes('test.mp4')


def test_save_scenes_csv(tmp_path):
    """Test saving scenes to CSV."""
    output_path = tmp_path / 'scenes.csv'
    
    scenes = [
        (0.0, 5.5),
        (5.5, 12.3),
        (12.3, 18.9),
    ]
    
    save_scenes_csv(scenes, str(output_path))
    
    assert output_path.exists()
    
    # Read back and verify
    with open(output_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 3
    
    # Check first scene
    assert rows[0]['scene_number'] == '1'
    assert rows[0]['start_time'] == '0.000'
    assert rows[0]['end_time'] == '5.500'
    assert rows[0]['duration'] == '5.500'
    
    # Check second scene
    assert rows[1]['scene_number'] == '2'
    assert rows[1]['start_time'] == '5.500'
    assert rows[1]['end_time'] == '12.300'
    assert rows[1]['duration'] == '6.800'


def test_save_scenes_csv_empty(tmp_path):
    """Test saving empty scenes list."""
    output_path = tmp_path / 'scenes.csv'
    
    save_scenes_csv([], str(output_path))
    
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 0


def test_save_scenes_csv_header(tmp_path):
    """Test CSV header format."""
    output_path = tmp_path / 'scenes.csv'
    
    save_scenes_csv([(0.0, 1.0)], str(output_path))
    
    with open(output_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    assert header == ['scene_number', 'start_time', 'end_time', 'duration']


def test_save_scenes_csv_formatting(tmp_path):
    """Test CSV float formatting (3 decimal places)."""
    output_path = tmp_path / 'scenes.csv'
    
    scenes = [
        (1.234567, 5.987654),
    ]
    
    save_scenes_csv(scenes, str(output_path))
    
    with open(output_path, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
    
    # Should be formatted to 3 decimal places
    assert row['start_time'] == '1.235'
    assert row['end_time'] == '5.988'
    assert row['duration'] == '4.753'


def test_detect_scenes_mocked():
    """Test detect_scenes with mocked dependencies."""
    mock_video_manager_cls = MagicMock()
    mock_scene_manager_cls = MagicMock()
    mock_content_detector_cls = MagicMock()
    
    with patch('gpu_video_tools.scenes.VideoManager', mock_video_manager_cls, create=True):
        with patch('gpu_video_tools.scenes.SceneManager', mock_scene_manager_cls, create=True):
            with patch('gpu_video_tools.scenes.ContentDetector', mock_content_detector_cls, create=True):
                with patch('gpu_video_tools.scenes.SCENEDETECT_AVAILABLE', True):
                    # Mock VideoManager
                    mock_video_manager = MagicMock()
                    mock_video_manager.get_base_timecode().framerate = 30
                    mock_video_manager_cls.return_value = mock_video_manager
                    
                    # Mock SceneManager
                    mock_scene_manager = MagicMock()
                    
                    # Mock scene list with timecodes
                    mock_timecode_start_1 = MagicMock()
                    mock_timecode_start_1.get_seconds.return_value = 0.0
                    mock_timecode_end_1 = MagicMock()
                    mock_timecode_end_1.get_seconds.return_value = 5.5
                    
                    mock_timecode_start_2 = MagicMock()
                    mock_timecode_start_2.get_seconds.return_value = 5.5
                    mock_timecode_end_2 = MagicMock()
                    mock_timecode_end_2.get_seconds.return_value = 12.3
                    
                    mock_scene_manager.get_scene_list.return_value = [
                        (mock_timecode_start_1, mock_timecode_end_1),
                        (mock_timecode_start_2, mock_timecode_end_2),
                    ]
                    
                    mock_scene_manager_cls.return_value = mock_scene_manager
                    
                    # Run detection
                    scenes = detect_scenes('test.mp4', threshold=25.0, min_scene_len=2.0)
                    
                    # Verify
                    assert len(scenes) == 2
                    assert scenes[0] == (0.0, 5.5)
                    assert scenes[1] == (5.5, 12.3)
                    
                    # Verify VideoManager was used correctly
                    mock_video_manager.start.assert_called_once()
                    mock_video_manager.release.assert_called_once()
                    
                    # Verify SceneManager was used correctly
                    mock_scene_manager.detect_scenes.assert_called_once()

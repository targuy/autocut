"""Scene detection wrapper using PySceneDetect."""

import csv
from pathlib import Path
from typing import List, Tuple

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: float = 1.0,
) -> List[Tuple[float, float]]:
    """Detect scene boundaries in a video.
    
    Args:
        video_path: Path to video file
        threshold: Detection threshold (default 27.0)
        min_scene_len: Minimum scene length in seconds (default 1.0)
    
    Returns:
        List of (start_time, end_time) tuples in seconds
    
    Raises:
        ImportError: If scenedetect is not installed
    """
    if not SCENEDETECT_AVAILABLE:
        raise ImportError("scenedetect is required. Install with: pip install scenedetect")
    
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * video_manager.get_base_timecode().framerate))
    )
    
    # Perform detection
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    # Get scene list
    scene_list = scene_manager.get_scene_list()
    
    # Convert to seconds
    scenes = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes.append((start_time, end_time))
    
    video_manager.release()
    
    return scenes


def save_scenes_csv(scenes: List[Tuple[float, float]], output_path: str):
    """Save scene boundaries to CSV.
    
    Args:
        scenes: List of (start_time, end_time) tuples
        output_path: Output CSV file path
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene_number', 'start_time', 'end_time', 'duration'])
        
        for i, (start, end) in enumerate(scenes, 1):
            duration = end - start
            writer.writerow([i, f'{start:.3f}', f'{end:.3f}', f'{duration:.3f}'])

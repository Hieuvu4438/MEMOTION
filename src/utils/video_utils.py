"""Video processing utilities."""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional, List
from pathlib import Path


class VideoReader:
    """Video reader with frame iteration support."""
    
    def __init__(self, source: str, target_fps: Optional[float] = None):
        """
        Initialize video reader.
        
        Args:
            source: Video file path or camera index
            target_fps: Target FPS for processing (None = use video FPS)
        """
        if isinstance(source, int) or source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
            self.is_camera = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_camera = False
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.target_fps = target_fps or self.fps
        self.frame_skip = max(1, int(self.fps / self.target_fps))
        
    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate over frames."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_skip == 0:
                yield frame_idx, frame
            
            frame_idx += 1
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read single frame."""
        return self.cap.read()
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get specific frame by index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release video capture."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Video writer for saving processed frames."""
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Output file path
            fps: Frames per second
            frame_size: (width, height)
            codec: Video codec
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")
    
    def write(self, frame: np.ndarray):
        """Write frame to video."""
        self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from video.
    
    Args:
        video_path: Input video path
        output_dir: Output directory for frames
        frame_interval: Extract every N-th frame
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of saved frame paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    with VideoReader(video_path) as reader:
        frame_count = 0
        for idx, frame in reader:
            if frame_interval > 1 and idx % frame_interval != 0:
                continue
            
            frame_path = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(str(frame_path))
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
    
    return saved_paths


def get_video_info(video_path: str) -> dict:
    """
    Get video information.
    
    Args:
        video_path: Video file path
    
    Returns:
        Dict with video info
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': 0.0,
    }
    
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info


def resize_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_dim: Optional[int] = None
) -> np.ndarray:
    """
    Resize frame maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_size: Target (width, height)
        max_dim: Maximum dimension
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if target_size:
        return cv2.resize(frame, target_size)
    
    if max_dim:
        scale = max_dim / max(h, w)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h))
    
    return frame


def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    fps: float = 30.0
):
    """
    Create video from list of frame images.
    
    Args:
        frame_paths: List of frame image paths
        output_path: Output video path
        fps: Frames per second
    """
    if not frame_paths:
        raise ValueError("No frames provided")
    
    first_frame = cv2.imread(frame_paths[0])
    h, w = first_frame.shape[:2]
    
    with VideoWriter(output_path, fps, (w, h)) as writer:
        for path in frame_paths:
            frame = cv2.imread(path)
            writer.write(frame)


def synchronize_videos(
    video1_path: str,
    video2_path: str,
    output_fps: float = 30.0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Synchronize two videos to same number of frames.
    Uses simple interpolation for different lengths.
    
    Args:
        video1_path: First video path
        video2_path: Second video path
        output_fps: Target FPS
    
    Returns:
        Tuple of (frames1, frames2) lists
    """
    frames1 = []
    frames2 = []
    
    with VideoReader(video1_path, target_fps=output_fps) as reader:
        for _, frame in reader:
            frames1.append(frame)
    
    with VideoReader(video2_path, target_fps=output_fps) as reader:
        for _, frame in reader:
            frames2.append(frame)
    
    # Match lengths by interpolation
    len1, len2 = len(frames1), len(frames2)
    
    if len1 != len2:
        target_len = max(len1, len2)
        
        if len1 < target_len:
            indices = np.linspace(0, len1 - 1, target_len).astype(int)
            frames1 = [frames1[i] for i in indices]
        
        if len2 < target_len:
            indices = np.linspace(0, len2 - 1, target_len).astype(int)
            frames2 = [frames2[i] for i in indices]
    
    return frames1, frames2
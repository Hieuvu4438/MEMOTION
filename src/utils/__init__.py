"""Utility functions."""

from .math_utils import (
    calculate_angle,
    calculate_distance,
    normalize_keypoints,
    cosine_similarity,
    weighted_cosine_similarity,
    oks_similarity,
    euclidean_distance_normalized,
    keypoints_to_angles,
    angle_difference,
    smooth_keypoints,
    calculate_body_bbox,
    get_body_orientation,
)

from .visualization import (
    draw_keypoints,
    draw_skeleton,
    draw_pose_comparison,
    draw_feedback_overlay,
    draw_angle_annotation,
    create_side_by_side_comparison,
    create_progress_bar,
    COLORS,
    SKELETON_COLORS,
)

from .video_utils import (
    VideoReader,
    VideoWriter,
    extract_frames,
    get_video_info,
    resize_frame,
    create_video_from_frames,
    synchronize_videos,
)

__all__ = [
    # Math
    'calculate_angle',
    'calculate_distance',
    'normalize_keypoints',
    'cosine_similarity',
    'weighted_cosine_similarity',
    'oks_similarity',
    'euclidean_distance_normalized',
    'keypoints_to_angles',
    'angle_difference',
    'smooth_keypoints',
    'calculate_body_bbox',
    'get_body_orientation',
    # Visualization
    'draw_keypoints',
    'draw_skeleton',
    'draw_pose_comparison',
    'draw_feedback_overlay',
    'draw_angle_annotation',
    'create_side_by_side_comparison',
    'create_progress_bar',
    'COLORS',
    'SKELETON_COLORS',
    # Video
    'VideoReader',
    'VideoWriter',
    'extract_frames',
    'get_video_info',
    'resize_frame',
    'create_video_from_frames',
    'synchronize_videos',
]
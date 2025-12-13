"""Mathematical utility functions for pose processing."""

import numpy as np
from typing import Tuple, List, Optional


def calculate_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray
) -> float:
    """
    Calculate angle at point_b formed by point_a -> point_b -> point_c.
    
    Args:
        point_a: First point [x, y]
        point_b: Vertex point [x, y]
        point_c: Third point [x, y]
    
    Returns:
        Angle in degrees (0-180)
    """
    ba = point_a - point_b
    bc = point_c - point_b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(point_a - point_b)


def normalize_keypoints(
    keypoints: np.ndarray,
    method: str = 'bbox'
) -> np.ndarray:
    """
    Normalize keypoints to make them scale and translation invariant.
    
    Args:
        keypoints: Array of shape (N, 2) or (N, 3) with [x, y] or [x, y, conf]
        method: Normalization method ('bbox', 'torso', 'minmax')
    
    Returns:
        Normalized keypoints
    """
    kpts = keypoints.copy()
    
    if kpts.shape[1] == 3:
        coords = kpts[:, :2]
        conf = kpts[:, 2:]
    else:
        coords = kpts
        conf = None
    
    valid_mask = ~np.any(np.isnan(coords), axis=1)
    if not np.any(valid_mask):
        return kpts
    
    valid_coords = coords[valid_mask]
    
    if method == 'bbox':
        min_coords = valid_coords.min(axis=0)
        max_coords = valid_coords.max(axis=0)
        scale = max_coords - min_coords
        scale[scale == 0] = 1
        coords = (coords - min_coords) / scale
        
    elif method == 'torso':
        # Normalize relative to torso (shoulders and hips)
        # Assuming COCO format: shoulders at 5,6 and hips at 11,12
        shoulder_center = (coords[5] + coords[6]) / 2
        hip_center = (coords[11] + coords[12]) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        if torso_length > 0:
            center = (shoulder_center + hip_center) / 2
            coords = (coords - center) / torso_length
    
    elif method == 'minmax':
        min_val = valid_coords.min()
        max_val = valid_coords.max()
        scale = max_val - min_val
        if scale > 0:
            coords = (coords - min_val) / scale
    
    if conf is not None:
        return np.hstack([coords, conf])
    return coords


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
    
    Returns:
        Cosine similarity value between -1 and 1
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def weighted_cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Calculate weighted cosine similarity.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        weights: Weight for each dimension
    
    Returns:
        Weighted cosine similarity
    """
    weighted_a = vec_a * weights
    weighted_b = vec_b * weights
    return cosine_similarity(weighted_a, weighted_b)


def oks_similarity(
    kpts_pred: np.ndarray,
    kpts_gt: np.ndarray,
    sigmas: np.ndarray,
    scale: float = 1.0
) -> float:
    """
    Calculate Object Keypoint Similarity (OKS).
    
    Args:
        kpts_pred: Predicted keypoints (N, 2) or (N, 3)
        kpts_gt: Ground truth keypoints (N, 2) or (N, 3)
        sigmas: Per-keypoint sigmas
        scale: Scale factor (typically bounding box area)
    
    Returns:
        OKS score between 0 and 1
    """
    if kpts_pred.shape[1] == 3:
        pred_coords = kpts_pred[:, :2]
        gt_coords = kpts_gt[:, :2]
        visibility = kpts_gt[:, 2] > 0
    else:
        pred_coords = kpts_pred
        gt_coords = kpts_gt
        visibility = np.ones(len(kpts_pred), dtype=bool)
    
    if not np.any(visibility):
        return 0.0
    
    # Calculate squared distances
    d_squared = np.sum((pred_coords - gt_coords) ** 2, axis=1)
    
    # OKS formula
    vars = (sigmas * 2) ** 2
    exp_term = np.exp(-d_squared / (2 * scale * vars + 1e-8))
    
    oks = np.sum(exp_term * visibility) / (np.sum(visibility) + 1e-8)
    
    return oks


def euclidean_distance_normalized(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray
) -> float:
    """
    Calculate normalized Euclidean distance between two pose keypoints.
    
    Args:
        kpts_a: First keypoints (N, 2)
        kpts_b: Second keypoints (N, 2)
    
    Returns:
        Normalized distance (0 = identical, higher = more different)
    """
    norm_a = normalize_keypoints(kpts_a, method='bbox')
    norm_b = normalize_keypoints(kpts_b, method='bbox')
    
    return np.mean(np.sqrt(np.sum((norm_a - norm_b) ** 2, axis=1)))


def keypoints_to_angles(
    keypoints: np.ndarray,
    joint_definitions: dict
) -> dict:
    """
    Convert keypoints to joint angles.
    
    Args:
        keypoints: Array of keypoints (N, 2) or (N, 3)
        joint_definitions: Dict mapping joint name to (idx_a, idx_b, idx_c)
    
    Returns:
        Dict mapping joint name to angle in degrees
    """
    coords = keypoints[:, :2] if keypoints.shape[1] == 3 else keypoints
    angles = {}
    
    for joint_name, (idx_a, idx_b, idx_c) in joint_definitions.items():
        try:
            angle = calculate_angle(coords[idx_a], coords[idx_b], coords[idx_c])
            angles[joint_name] = angle
        except (IndexError, ValueError):
            angles[joint_name] = None
    
    return angles


def angle_difference(angle_a: float, angle_b: float) -> float:
    """
    Calculate the absolute difference between two angles.
    Handles wraparound for angles near 180 degrees.
    """
    diff = abs(angle_a - angle_b)
    return min(diff, 360 - diff)


def smooth_keypoints(
    keypoints_sequence: List[np.ndarray],
    window_size: int = 5,
    method: str = 'moving_average'
) -> List[np.ndarray]:
    """
    Smooth keypoints over time to reduce jitter.
    
    Args:
        keypoints_sequence: List of keypoints arrays
        window_size: Smoothing window size
        method: 'moving_average' or 'exponential'
    
    Returns:
        Smoothed keypoints sequence
    """
    if len(keypoints_sequence) < window_size:
        return keypoints_sequence
    
    smoothed = []
    stack = np.stack(keypoints_sequence, axis=0)
    
    if method == 'moving_average':
        for i in range(len(keypoints_sequence)):
            start = max(0, i - window_size // 2)
            end = min(len(keypoints_sequence), i + window_size // 2 + 1)
            smoothed.append(np.mean(stack[start:end], axis=0))
    
    elif method == 'exponential':
        alpha = 2 / (window_size + 1)
        smoothed.append(keypoints_sequence[0])
        for i in range(1, len(keypoints_sequence)):
            smooth_kpt = alpha * keypoints_sequence[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smooth_kpt)
    
    return smoothed


def calculate_body_bbox(keypoints: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box from keypoints.
    
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    coords = keypoints[:, :2] if keypoints.shape[1] == 3 else keypoints
    valid = ~np.any(np.isnan(coords), axis=1)
    
    if not np.any(valid):
        return (0, 0, 0, 0)
    
    valid_coords = coords[valid]
    x_min, y_min = valid_coords.min(axis=0).astype(int)
    x_max, y_max = valid_coords.max(axis=0).astype(int)
    
    return (x_min, y_min, x_max, y_max)


def get_body_orientation(keypoints: np.ndarray) -> str:
    """
    Determine if the person is facing front, back, left, or right.
    
    Args:
        keypoints: COCO format keypoints
    
    Returns:
        'front', 'back', 'left', 'right'
    """
    coords = keypoints[:, :2] if keypoints.shape[1] == 3 else keypoints
    
    left_shoulder = coords[5]
    right_shoulder = coords[6]
    nose = coords[0]
    
    shoulder_center = (left_shoulder + right_shoulder) / 2
    shoulder_width = np.abs(left_shoulder[0] - right_shoulder[0])
    
    # Check if shoulders are roughly horizontal (front/back view)
    if shoulder_width > 0.3 * np.linalg.norm(left_shoulder - right_shoulder):
        # Check nose position relative to shoulders
        if nose[1] < shoulder_center[1]:
            return 'front'
        else:
            return 'back'
    else:
        # Side view
        if left_shoulder[0] < right_shoulder[0]:
            return 'right'
        else:
            return 'left'
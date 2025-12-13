"""Visualization utilities for pose estimation."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from ..config.pose_config import SKELETON_CONNECTIONS, COCO_KEYPOINTS


# Color scheme
COLORS = {
    'skeleton': (0, 255, 0),       # Green
    'keypoint': (255, 0, 0),       # Blue
    'error': (0, 0, 255),          # Red
    'correct': (0, 255, 0),        # Green
    'warning': (0, 165, 255),      # Orange
    'text': (255, 255, 255),       # White
    'background': (0, 0, 0),       # Black
}

# Skeleton colors for different body parts
SKELETON_COLORS = {
    'head': (255, 255, 0),      # Cyan
    'torso': (255, 0, 255),     # Magenta
    'left_arm': (0, 255, 0),    # Green
    'right_arm': (0, 255, 255), # Yellow
    'left_leg': (255, 128, 0),  # Blue-ish
    'right_leg': (0, 128, 255), # Orange-ish
}


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5,
    radius: int = 5,
    color: Tuple[int, int, int] = None
) -> np.ndarray:
    """
    Draw keypoints on image.
    
    Args:
        image: Input image
        keypoints: Keypoints array (N, 2) or (N, 3)
        scores: Confidence scores (optional if keypoints has 3 columns)
        confidence_threshold: Minimum confidence to draw
        radius: Circle radius
        color: Override color for all keypoints
    
    Returns:
        Image with keypoints drawn
    """
    img = image.copy()
    
    if keypoints.shape[1] == 3:
        coords = keypoints[:, :2]
        scores = keypoints[:, 2] if scores is None else scores
    else:
        coords = keypoints
        scores = np.ones(len(keypoints)) if scores is None else scores
    
    for i, (pt, score) in enumerate(zip(coords, scores)):
        if score < confidence_threshold:
            continue
        
        x, y = int(pt[0]), int(pt[1])
        if x < 0 or y < 0:
            continue
        
        pt_color = color if color else COLORS['keypoint']
        cv2.circle(img, (x, y), radius, pt_color, -1)
        
    return img


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5,
    line_width: int = 2,
    colored: bool = True
) -> np.ndarray:
    """
    Draw skeleton connections on image.
    
    Args:
        image: Input image
        keypoints: Keypoints array (N, 2) or (N, 3)
        scores: Confidence scores
        confidence_threshold: Minimum confidence
        line_width: Line width
        colored: Use different colors for body parts
    
    Returns:
        Image with skeleton drawn
    """
    img = image.copy()
    
    if keypoints.shape[1] == 3:
        coords = keypoints[:, :2]
        scores = keypoints[:, 2] if scores is None else scores
    else:
        coords = keypoints
        scores = np.ones(len(keypoints)) if scores is None else scores
    
    # Define which connections belong to which body part
    connection_parts = {
        (0, 1): 'head', (0, 2): 'head', (1, 3): 'head', (2, 4): 'head',
        (5, 6): 'torso', (5, 11): 'torso', (6, 12): 'torso', (11, 12): 'torso',
        (5, 7): 'left_arm', (7, 9): 'left_arm',
        (6, 8): 'right_arm', (8, 10): 'right_arm',
        (11, 13): 'left_leg', (13, 15): 'left_leg',
        (12, 14): 'right_leg', (14, 16): 'right_leg',
    }
    
    for connection in SKELETON_CONNECTIONS:
        idx1, idx2 = connection
        
        if scores[idx1] < confidence_threshold or scores[idx2] < confidence_threshold:
            continue
        
        pt1 = (int(coords[idx1][0]), int(coords[idx1][1]))
        pt2 = (int(coords[idx2][0]), int(coords[idx2][1]))
        
        if pt1[0] < 0 or pt1[1] < 0 or pt2[0] < 0 or pt2[1] < 0:
            continue
        
        if colored:
            part = connection_parts.get(connection, 'torso')
            color = SKELETON_COLORS.get(part, COLORS['skeleton'])
        else:
            color = COLORS['skeleton']
        
        cv2.line(img, pt1, pt2, color, line_width)
    
    return img


def draw_pose_comparison(
    image: np.ndarray,
    user_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    error_joints: List[str] = None,
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw both user and reference poses for comparison.
    
    Args:
        image: Input image
        user_keypoints: User's keypoints
        reference_keypoints: Reference keypoints (will be scaled to user's position)
        error_joints: List of joint names that have errors
        confidence_threshold: Minimum confidence
    
    Returns:
        Image with both poses drawn
    """
    from ..config.pose_config import KEYPOINT_INDEX
    
    img = image.copy()
    
    # Draw user skeleton in green
    img = draw_skeleton(img, user_keypoints, confidence_threshold=confidence_threshold)
    img = draw_keypoints(img, user_keypoints, confidence_threshold=confidence_threshold)
    
    # Highlight error joints in red
    if error_joints:
        user_coords = user_keypoints[:, :2] if user_keypoints.shape[1] == 3 else user_keypoints
        for joint in error_joints:
            if joint in KEYPOINT_INDEX:
                idx = KEYPOINT_INDEX[joint]
                pt = user_coords[idx]
                cv2.circle(img, (int(pt[0]), int(pt[1])), 10, COLORS['error'], 3)
    
    return img


def draw_feedback_overlay(
    image: np.ndarray,
    feedback_messages: List[str],
    score: float,
    position: str = 'top'
) -> np.ndarray:
    """
    Draw feedback messages on image.
    
    Args:
        image: Input image
        feedback_messages: List of feedback strings
        score: Overall similarity score (0-100)
        position: 'top' or 'bottom'
    
    Returns:
        Image with feedback overlay
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Create semi-transparent overlay
    overlay = img.copy()
    
    # Calculate text area
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 25
    padding = 10
    
    n_lines = len(feedback_messages) + 1  # +1 for score
    box_height = n_lines * line_height + 2 * padding
    
    if position == 'top':
        box_y = 0
    else:
        box_y = h - box_height
    
    # Draw background box
    cv2.rectangle(overlay, (0, box_y), (w, box_y + box_height), COLORS['background'], -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    
    # Draw score
    score_color = COLORS['correct'] if score >= 85 else (COLORS['warning'] if score >= 70 else COLORS['error'])
    score_text = f"Score: {score:.1f}%"
    cv2.putText(img, score_text, (padding, box_y + padding + line_height), 
                font, font_scale * 1.2, score_color, thickness + 1)
    
    # Draw feedback messages
    for i, msg in enumerate(feedback_messages[:5]):  # Max 5 messages
        y = box_y + padding + (i + 2) * line_height
        cv2.putText(img, f"• {msg}", (padding, y), font, font_scale, COLORS['text'], thickness)
    
    return img


def draw_angle_annotation(
    image: np.ndarray,
    keypoints: np.ndarray,
    joint_name: str,
    angle: float,
    joint_indices: Tuple[int, int, int],
    is_error: bool = False
) -> np.ndarray:
    """
    Draw angle annotation at a joint.
    
    Args:
        image: Input image
        keypoints: Keypoints array
        joint_name: Name of the joint
        angle: Angle value
        joint_indices: (idx_a, idx_b, idx_c) - angle is at idx_b
        is_error: Whether this joint has an error
    
    Returns:
        Image with angle annotation
    """
    img = image.copy()
    coords = keypoints[:, :2] if keypoints.shape[1] == 3 else keypoints
    
    idx_a, idx_b, idx_c = joint_indices
    pt_b = coords[idx_b]
    
    color = COLORS['error'] if is_error else COLORS['correct']
    
    # Draw arc
    # (simplified - just draw text for now)
    text = f"{angle:.0f}°"
    cv2.putText(img, text, (int(pt_b[0]) + 10, int(pt_b[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img


def create_side_by_side_comparison(
    user_image: np.ndarray,
    reference_image: np.ndarray,
    user_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    labels: Tuple[str, str] = ("User", "Reference")
) -> np.ndarray:
    """
    Create side-by-side comparison of user and reference poses.
    
    Args:
        user_image: User's image
        reference_image: Reference image
        user_keypoints: User's keypoints
        reference_keypoints: Reference keypoints
        labels: Labels for each image
    
    Returns:
        Combined image
    """
    # Resize images to same height
    h1, w1 = user_image.shape[:2]
    h2, w2 = reference_image.shape[:2]
    
    target_h = max(h1, h2)
    
    if h1 != target_h:
        scale = target_h / h1
        user_image = cv2.resize(user_image, None, fx=scale, fy=scale)
        user_keypoints = user_keypoints.copy()
        user_keypoints[:, :2] *= scale
    
    if h2 != target_h:
        scale = target_h / h2
        reference_image = cv2.resize(reference_image, None, fx=scale, fy=scale)
        reference_keypoints = reference_keypoints.copy()
        reference_keypoints[:, :2] *= scale
    
    # Draw poses
    user_vis = draw_skeleton(user_image, user_keypoints)
    user_vis = draw_keypoints(user_vis, user_keypoints)
    
    ref_vis = draw_skeleton(reference_image, reference_keypoints)
    ref_vis = draw_keypoints(ref_vis, reference_keypoints)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(user_vis, labels[0], (10, 30), font, 1, COLORS['text'], 2)
    cv2.putText(ref_vis, labels[1], (10, 30), font, 1, COLORS['text'], 2)
    
    # Combine
    combined = np.hstack([user_vis, ref_vis])
    
    return combined


def create_progress_bar(
    width: int,
    height: int,
    progress: float,
    label: str = ""
) -> np.ndarray:
    """
    Create a progress bar image.
    
    Args:
        width: Bar width
        height: Bar height
        progress: Progress value (0-1)
        label: Optional label
    
    Returns:
        Progress bar image
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background
    cv2.rectangle(img, (0, 0), (width-1, height-1), (50, 50, 50), -1)
    cv2.rectangle(img, (0, 0), (width-1, height-1), (100, 100, 100), 1)
    
    # Progress
    progress_width = int((width - 4) * min(progress, 1.0))
    color = COLORS['correct'] if progress >= 0.85 else (COLORS['warning'] if progress >= 0.70 else COLORS['error'])
    cv2.rectangle(img, (2, 2), (2 + progress_width, height-3), color, -1)
    
    # Label
    if label:
        cv2.putText(img, label, (5, height//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
    
    return img
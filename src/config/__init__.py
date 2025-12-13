"""Configuration module."""

from .model_config import get_model_config, RTMPOSE_CONFIGS, RTMPoseConfig
from .pose_config import (
    COCO_KEYPOINTS,
    KEYPOINT_INDEX,
    SKELETON_CONNECTIONS,
    JOINT_ANGLES,
    BODY_PARTS,
    KEYPOINT_WEIGHTS,
    OKS_SIGMAS,
    YOGA_POSES,
    PHYSICAL_THERAPY_EXERCISES,
    ERROR_THRESHOLDS,
    FEEDBACK_MESSAGES,
)

__all__ = [
    'get_model_config',
    'RTMPOSE_CONFIGS',
    'RTMPoseConfig',
    'COCO_KEYPOINTS',
    'KEYPOINT_INDEX',
    'SKELETON_CONNECTIONS',
    'JOINT_ANGLES',
    'BODY_PARTS',
    'KEYPOINT_WEIGHTS',
    'OKS_SIGMAS',
    'YOGA_POSES',
    'PHYSICAL_THERAPY_EXERCISES',
    'ERROR_THRESHOLDS',
    'FEEDBACK_MESSAGES',
]
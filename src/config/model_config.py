"""Model configurations for RTMPose variants."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RTMPoseConfig:
    """Configuration for RTMPose model."""
    name: str
    input_size: Tuple[int, int]
    num_keypoints: int
    onnx_url: str
    detector_url: str
    ap_coco: float
    params_m: float
    recommended_for: str


RTMPOSE_CONFIGS = {
    't': RTMPoseConfig(
        name='rtmpose-t',
        input_size=(192, 256),
        num_keypoints=17,
        onnx_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        detector_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
        ap_coco=68.5,
        params_m=3.3,
        recommended_for='IoT devices, extremely low power'
    ),
    's': RTMPoseConfig(
        name='rtmpose-s',
        input_size=(192, 256),
        num_keypoints=17,
        onnx_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',
        detector_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-40f1f3a9.zip',
        ap_coco=72.2,
        params_m=5.5,
        recommended_for='Mobile devices (recommended)'
    ),
    'm': RTMPoseConfig(
        name='rtmpose-m',
        input_size=(192, 256),
        num_keypoints=17,
        onnx_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
        detector_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
        ap_coco=75.8,
        params_m=13.6,
        recommended_for='Desktop, balanced accuracy/speed'
    ),
    'l': RTMPoseConfig(
        name='rtmpose-l',
        input_size=(256, 320),
        num_keypoints=17,
        onnx_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip',
        detector_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a6a.zip',
        ap_coco=76.5,
        params_m=27.6,
        recommended_for='High accuracy requirements'
    )
}


def get_model_config(size: str = 's') -> RTMPoseConfig:
    """Get model configuration by size."""
    if size not in RTMPOSE_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(RTMPOSE_CONFIGS.keys())}")
    return RTMPOSE_CONFIGS[size]
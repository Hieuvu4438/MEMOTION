"""Pydantic schemas for API."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


class PoseName(str, Enum):
    """Available yoga poses."""
    WARRIOR_I = "warrior_i"
    WARRIOR_II = "warrior_ii"
    TREE_POSE = "tree_pose"
    DOWNWARD_DOG = "downward_dog"
    COBRA_POSE = "cobra_pose"


class ComparisonMethod(str, Enum):
    """Comparison methods."""
    COSINE = "cosine"
    ANGLE = "angle"
    OKS = "oks"
    WEIGHTED = "weighted"


class Keypoint(BaseModel):
    """Single keypoint."""
    x: float
    y: float
    confidence: float = 1.0


class PoseKeypoints(BaseModel):
    """Full pose keypoints."""
    keypoints: List[Keypoint]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "keypoints": [
                    {"x": 100, "y": 50, "confidence": 0.95},
                ]
            }
        }
    }


class JointErrorSchema(BaseModel):
    """Joint error detail."""
    joint_name: str
    error_type: str
    current_angle: float
    expected_angle: float
    difference: float
    severity: str
    suggestion: str


class FeedbackSchema(BaseModel):
    """Feedback response."""
    overall_score: float
    rating: str
    errors: List[JointErrorSchema]
    suggestions: List[str]
    positive_feedback: List[str]
    pose_name: Optional[str] = None


class AssessmentRequest(BaseModel):
    """Request for pose assessment."""
    pose_name: Optional[PoseName] = None
    comparison_method: ComparisonMethod = ComparisonMethod.WEIGHTED
    return_visualization: bool = False
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "pose_name": "warrior_i",
                "comparison_method": "weighted",
                "return_visualization": True
            }
        }
    }


class AssessmentResponse(BaseModel):
    """Response for pose assessment."""
    success: bool
    overall_score: float
    feedback: FeedbackSchema
    joint_scores: Dict[str, float]
    error_joints: List[str]
    visualization_base64: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "overall_score": 85.5,
                "feedback": {
                    "overall_score": 85.5,
                    "rating": "good",
                    "errors": [],
                    "suggestions": ["Giữ vững tư thế!"],
                    "positive_feedback": ["Vai trái đã đúng vị trí"]
                },
                "joint_scores": {"left_shoulder": 90.0},
                "error_joints": []
            }
        }
    }


class VideoAssessmentRequest(BaseModel):
    """Request for video assessment."""
    use_dtw: bool = True
    generate_comparison_video: bool = False


class VideoAssessmentResponse(BaseModel):
    """Response for video assessment."""
    success: bool
    overall_score: float
    frame_scores: List[float]
    joint_scores: Dict[str, float]
    error_joints: List[str]
    best_frame: int
    worst_frame: int
    comparison_video_url: Optional[str] = None


class RealtimeFeedbackResponse(BaseModel):
    """Response for real-time feedback."""
    score: float
    messages: List[str]
    keypoints: List[Keypoint]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
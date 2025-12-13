"""Main pose assessment logic."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass, field

from ..models.rtmpose_wrapper import RTMPoseWrapper, SimplePoseEstimator
from ..config.pose_config import YOGA_POSES, ERROR_THRESHOLDS
from .pose_comparator import PoseComparator, ComparisonResult
from .feedback_generator import FeedbackGenerator, FeedbackResult


@dataclass
class AssessmentResult:
    """Complete assessment result."""
    overall_score: float
    comparison: ComparisonResult
    feedback: FeedbackResult
    user_keypoints: np.ndarray
    reference_keypoints: np.ndarray
    user_image: Optional[np.ndarray] = None
    reference_image: Optional[np.ndarray] = None
    visualized_image: Optional[np.ndarray] = None


class PoseAssessor:
    """
    Main class for pose assessment.
    Combines pose estimation, comparison, and feedback generation.
    """
    
    def __init__(
        self,
        model_size: str = 's',
        device: str = 'cpu',
        comparison_method: str = 'weighted',
        language: str = 'vi',
        weights_dir: str = 'weights'
    ):
        """
        Initialize pose assessor.
        
        Args:
            model_size: RTMPose model size ('t', 's', 'm', 'l')
            device: 'cpu' or 'cuda'
            comparison_method: Comparison method for PoseComparator
            language: Feedback language ('vi' or 'en')
            weights_dir: Directory for model weights
        """
        self.model_size = model_size
        self.device = device
        
        # Initialize components
        self.pose_estimator = SimplePoseEstimator(
            size=model_size,
            device=device,
            weights_dir=weights_dir
        )
        
        self.comparator = PoseComparator(method=comparison_method)
        self.feedback_generator = FeedbackGenerator(language=language)
        
        # Cache for reference poses
        self._reference_cache = {}
    
    def estimate_pose(
        self,
        image: Union[str, np.ndarray],
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Estimate pose from image.
        
        Args:
            image: Image path or numpy array (BGR)
            confidence_threshold: Minimum keypoint confidence
        
        Returns:
            Keypoints array (17, 3) with [x, y, confidence]
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image: {image}")
        
        return self.pose_estimator.estimate_with_confidence(image, confidence_threshold)
    
    def compare_poses(
        self,
        user_keypoints: np.ndarray,
        reference_keypoints: np.ndarray
    ) -> ComparisonResult:
        """
        Compare two poses.
        
        Args:
            user_keypoints: User's keypoints
            reference_keypoints: Reference keypoints
        
        Returns:
            ComparisonResult
        """
        return self.comparator.compare(user_keypoints, reference_keypoints)
    
    def assess(
        self,
        user_image: Union[str, np.ndarray],
        reference_image: Union[str, np.ndarray],
        pose_name: Optional[str] = None,
        visualize: bool = False
    ) -> AssessmentResult:
        """
        Full assessment: estimate poses, compare, generate feedback.
        
        Args:
            user_image: User's image
            reference_image: Reference image
            pose_name: Name of pose for specific feedback
            visualize: Whether to generate visualization
        
        Returns:
            AssessmentResult with all details
        """
        # Load images if paths
        if isinstance(user_image, str):
            user_img = cv2.imread(user_image)
        else:
            user_img = user_image.copy()
        
        if isinstance(reference_image, str):
            ref_img = cv2.imread(reference_image)
        else:
            ref_img = reference_image.copy()
        
        # Estimate poses
        user_kpts = self.estimate_pose(user_img)
        ref_kpts = self.estimate_pose(ref_img)
        
        # Compare
        comparison = self.compare_poses(user_kpts, ref_kpts)
        
        # Generate feedback
        feedback = self.feedback_generator.generate_feedback(
            comparison, user_kpts, ref_kpts, pose_name
        )
        
        # Visualization
        vis_image = None
        if visualize:
            vis_image = self._create_visualization(
                user_img, user_kpts, ref_kpts, comparison, feedback
            )
        
        return AssessmentResult(
            overall_score=comparison.overall_score,
            comparison=comparison,
            feedback=feedback,
            user_keypoints=user_kpts,
            reference_keypoints=ref_kpts,
            user_image=user_img,
            reference_image=ref_img,
            visualized_image=vis_image
        )
    
    def assess_with_reference(
        self,
        user_image: Union[str, np.ndarray],
        reference_keypoints: np.ndarray,
        pose_name: Optional[str] = None,
        visualize: bool = False
    ) -> AssessmentResult:
        """
        Assess using pre-computed reference keypoints.
        Useful for real-time assessment with cached reference.
        
        Args:
            user_image: User's image
            reference_keypoints: Pre-computed reference keypoints
            pose_name: Name of pose
            visualize: Whether to visualize
        
        Returns:
            AssessmentResult
        """
        if isinstance(user_image, str):
            user_img = cv2.imread(user_image)
        else:
            user_img = user_image.copy()
        
        user_kpts = self.estimate_pose(user_img)
        comparison = self.compare_poses(user_kpts, reference_keypoints)
        feedback = self.feedback_generator.generate_feedback(
            comparison, user_kpts, reference_keypoints, pose_name
        )
        
        vis_image = None
        if visualize:
            vis_image = self._create_visualization(
                user_img, user_kpts, reference_keypoints, comparison, feedback
            )
        
        return AssessmentResult(
            overall_score=comparison.overall_score,
            comparison=comparison,
            feedback=feedback,
            user_keypoints=user_kpts,
            reference_keypoints=reference_keypoints,
            user_image=user_img,
            visualized_image=vis_image
        )
    
    def load_reference(
        self,
        reference_source: Union[str, np.ndarray],
        name: str
    ) -> np.ndarray:
        """
        Load and cache reference pose.
        
        Args:
            reference_source: Image path or numpy array
            name: Name for caching
        
        Returns:
            Reference keypoints
        """
        if name in self._reference_cache:
            return self._reference_cache[name]
        
        keypoints = self.estimate_pose(reference_source)
        self._reference_cache[name] = keypoints
        
        return keypoints
    
    def _create_visualization(
        self,
        user_image: np.ndarray,
        user_kpts: np.ndarray,
        ref_kpts: np.ndarray,
        comparison: ComparisonResult,
        feedback: FeedbackResult
    ) -> np.ndarray:
        """Create visualization image."""
        from ..utils.visualization import (
            draw_skeleton,
            draw_keypoints,
            draw_feedback_overlay,
            COLORS
        )
        
        img = user_image.copy()
        
        # Draw skeleton
        img = draw_skeleton(img, user_kpts)
        img = draw_keypoints(img, user_kpts)
        
        # Highlight errors
        for joint in comparison.error_joints:
            from ..config.pose_config import KEYPOINT_INDEX
            if joint in KEYPOINT_INDEX:
                idx = KEYPOINT_INDEX[joint]
                pt = user_kpts[idx, :2].astype(int)
                cv2.circle(img, tuple(pt), 12, COLORS['error'], 3)
        
        # Add feedback overlay
        messages = [e.suggestion for e in feedback.errors[:3]]
        img = draw_feedback_overlay(img, messages, comparison.overall_score)
        
        return img
    
    def assess_yoga_pose(
        self,
        user_image: Union[str, np.ndarray],
        pose_name: str,
        reference_image: Optional[Union[str, np.ndarray]] = None,
        visualize: bool = True
    ) -> AssessmentResult:
        """
        Assess a specific yoga pose.
        
        Args:
            user_image: User's image
            pose_name: Name of yoga pose (from YOGA_POSES)
            reference_image: Optional reference image
            visualize: Whether to visualize
        
        Returns:
            AssessmentResult with yoga-specific feedback
        """
        if pose_name not in YOGA_POSES:
            raise ValueError(f"Unknown yoga pose: {pose_name}")
        
        if reference_image is not None:
            return self.assess(
                user_image, reference_image,
                pose_name=pose_name, visualize=visualize
            )
        
        # Use cached reference or require reference
        if pose_name in self._reference_cache:
            return self.assess_with_reference(
                user_image, self._reference_cache[pose_name],
                pose_name=pose_name, visualize=visualize
            )
        
        raise ValueError(f"No reference available for pose: {pose_name}. "
                        "Please provide reference_image or load_reference() first.")
    
    def get_realtime_feedback(
        self,
        user_image: np.ndarray,
        reference_keypoints: np.ndarray,
        max_messages: int = 2
    ) -> Tuple[float, List[str], np.ndarray]:
        """
        Get quick feedback for real-time display.
        
        Args:
            user_image: Current frame
            reference_keypoints: Reference pose keypoints
            max_messages: Max feedback messages
        
        Returns:
            (score, messages, user_keypoints)
        """
        user_kpts = self.estimate_pose(user_image)
        comparison = self.compare_poses(user_kpts, reference_keypoints)
        messages = self.feedback_generator.generate_realtime_feedback(
            comparison, max_messages
        )
        
        return comparison.overall_score, messages, user_kpts


class BatchAssessor:
    """Batch assessment for multiple images or video frames."""
    
    def __init__(self, assessor: PoseAssessor):
        """
        Initialize batch assessor.
        
        Args:
            assessor: PoseAssessor instance
        """
        self.assessor = assessor
    
    def assess_batch(
        self,
        user_images: List[Union[str, np.ndarray]],
        reference_image: Union[str, np.ndarray],
        pose_name: Optional[str] = None
    ) -> List[AssessmentResult]:
        """
        Assess multiple images against single reference.
        
        Args:
            user_images: List of user images
            reference_image: Reference image
            pose_name: Pose name for feedback
        
        Returns:
            List of AssessmentResults
        """
        # Pre-compute reference
        ref_kpts = self.assessor.estimate_pose(reference_image)
        
        results = []
        for img in user_images:
            result = self.assessor.assess_with_reference(
                img, ref_kpts, pose_name, visualize=False
            )
            results.append(result)
        
        return results
    
    def generate_session_report(
        self,
        results: List[AssessmentResult],
        exercise_name: str
    ) -> dict:
        """
        Generate summary report from batch results.
        
        Args:
            results: List of assessment results
            exercise_name: Name of exercise
        
        Returns:
            Session summary dict
        """
        feedbacks = [r.feedback for r in results]
        return self.assessor.feedback_generator.generate_exercise_report(
            feedbacks, exercise_name
        )
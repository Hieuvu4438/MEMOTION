"""Feedback generation for pose assessment."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..config.pose_config import (
    JOINT_ANGLES,
    YOGA_POSES,
    PHYSICAL_THERAPY_EXERCISES,
    ERROR_THRESHOLDS,
    FEEDBACK_MESSAGES,
    COCO_KEYPOINTS,
)
from ..utils.math_utils import keypoints_to_angles, angle_difference
from .pose_comparator import ComparisonResult


@dataclass
class JointError:
    """Detailed error information for a joint."""
    joint_name: str
    error_type: str  # 'too_bent', 'too_straight', 'misaligned', etc.
    current_angle: float
    expected_angle: float
    difference: float
    severity: str  # 'minor', 'moderate', 'severe'
    suggestion: str


@dataclass
class FeedbackResult:
    """Complete feedback for a pose assessment."""
    overall_score: float
    rating: str  # 'excellent', 'good', 'needs_improvement', 'poor'
    errors: List[JointError] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    positive_feedback: List[str] = field(default_factory=list)
    pose_name: Optional[str] = None


class FeedbackGenerator:
    """Generate human-readable feedback for pose assessment."""
    
    # Rating thresholds
    RATING_THRESHOLDS = {
        'excellent': 90,
        'good': 75,
        'needs_improvement': 60,
        'poor': 0
    }
    
    # Severity thresholds (angle difference in degrees)
    SEVERITY_THRESHOLDS = {
        'minor': 10,
        'moderate': 20,
        'severe': 30
    }
    
    def __init__(self, language: str = 'vi'):
        """
        Initialize feedback generator.
        
        Args:
            language: 'vi' for Vietnamese, 'en' for English
        """
        self.language = language
        self.feedback_templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load feedback templates."""
        if self.language == 'vi':
            return {
                'excellent': 'Xuất sắc! Tư thế của bạn rất chuẩn.',
                'good': 'Tốt! Chỉ cần điều chỉnh nhẹ.',
                'needs_improvement': 'Cần cải thiện một số điểm.',
                'poor': 'Cần điều chỉnh nhiều. Hãy xem gợi ý bên dưới.',
                'joint_correct': '{joint} đã đúng vị trí.',
                'too_bent': 'Duỗi thẳng {joint} hơn (hiện tại: {current}°, cần: {expected}°)',
                'too_straight': 'Gập {joint} hơn (hiện tại: {current}°, cần: {expected}°)',
                'misaligned': 'Điều chỉnh {joint} cho thẳng hàng hơn',
            }
        else:
            return {
                'excellent': 'Excellent! Your pose is very accurate.',
                'good': 'Good! Just minor adjustments needed.',
                'needs_improvement': 'Some improvements needed.',
                'poor': 'Needs significant adjustment. See suggestions below.',
                'joint_correct': '{joint} is in correct position.',
                'too_bent': 'Straighten your {joint} more (current: {current}°, target: {expected}°)',
                'too_straight': 'Bend your {joint} more (current: {current}°, target: {expected}°)',
                'misaligned': 'Adjust your {joint} for better alignment',
            }
    
    def generate_feedback(
        self,
        comparison_result: ComparisonResult,
        user_keypoints: 'np.ndarray',
        reference_keypoints: 'np.ndarray',
        pose_name: Optional[str] = None
    ) -> FeedbackResult:
        """
        Generate detailed feedback from comparison result.
        
        Args:
            comparison_result: Result from PoseComparator
            user_keypoints: User's keypoints
            reference_keypoints: Reference keypoints
            pose_name: Name of the pose (for specific feedback)
        
        Returns:
            FeedbackResult with detailed feedback
        """
        score = comparison_result.overall_score
        
        # Determine rating
        rating = self._get_rating(score)
        
        # Analyze errors
        errors = self._analyze_errors(
            comparison_result,
            user_keypoints,
            reference_keypoints,
            pose_name
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(errors, pose_name)
        
        # Generate positive feedback
        positive = self._generate_positive_feedback(comparison_result)
        
        return FeedbackResult(
            overall_score=score,
            rating=rating,
            errors=errors,
            suggestions=suggestions,
            positive_feedback=positive,
            pose_name=pose_name
        )
    
    def _get_rating(self, score: float) -> str:
        """Get rating based on score."""
        for rating, threshold in self.RATING_THRESHOLDS.items():
            if score >= threshold:
                return rating
        return 'poor'
    
    def _analyze_errors(
        self,
        result: ComparisonResult,
        user_kpts: 'np.ndarray',
        ref_kpts: 'np.ndarray',
        pose_name: Optional[str]
    ) -> List[JointError]:
        """Analyze joint errors in detail."""
        import numpy as np
        
        errors = []
        
        # Get angles
        user_angles = keypoints_to_angles(user_kpts, JOINT_ANGLES)
        ref_angles = keypoints_to_angles(ref_kpts, JOINT_ANGLES)
        
        # Check each error joint
        for joint_name in result.error_joints:
            if joint_name not in user_angles or joint_name not in ref_angles:
                continue
            
            user_angle = user_angles[joint_name]
            ref_angle = ref_angles[joint_name]
            
            if user_angle is None or ref_angle is None:
                continue
            
            diff = angle_difference(user_angle, ref_angle)
            
            # Determine error type
            if user_angle < ref_angle - 5:
                error_type = 'too_bent'
            elif user_angle > ref_angle + 5:
                error_type = 'too_straight'
            else:
                error_type = 'misaligned'
            
            # Determine severity
            severity = self._get_severity(diff)
            
            # Generate suggestion
            suggestion = self._format_suggestion(joint_name, error_type, user_angle, ref_angle)
            
            errors.append(JointError(
                joint_name=joint_name,
                error_type=error_type,
                current_angle=user_angle,
                expected_angle=ref_angle,
                difference=diff,
                severity=severity,
                suggestion=suggestion
            ))
        
        # Sort by severity
        severity_order = {'severe': 0, 'moderate': 1, 'minor': 2}
        errors.sort(key=lambda e: severity_order.get(e.severity, 3))
        
        return errors
    
    def _get_severity(self, angle_diff: float) -> str:
        """Get severity level based on angle difference."""
        if angle_diff >= self.SEVERITY_THRESHOLDS['severe']:
            return 'severe'
        elif angle_diff >= self.SEVERITY_THRESHOLDS['moderate']:
            return 'moderate'
        else:
            return 'minor'
    
    def _format_suggestion(
        self,
        joint_name: str,
        error_type: str,
        current: float,
        expected: float
    ) -> str:
        """Format suggestion message."""
        # Get joint display name
        joint_display = self._get_joint_display_name(joint_name)
        
        # Use template
        template = self.feedback_templates.get(error_type, self.feedback_templates['misaligned'])
        
        return template.format(
            joint=joint_display,
            current=f"{current:.0f}",
            expected=f"{expected:.0f}"
        )
    
    def _get_joint_display_name(self, joint_name: str) -> str:
        """Get display name for joint."""
        if self.language == 'vi':
            names = {
                'left_elbow': 'khuỷu tay trái',
                'right_elbow': 'khuỷu tay phải',
                'left_shoulder': 'vai trái',
                'right_shoulder': 'vai phải',
                'left_hip': 'hông trái',
                'right_hip': 'hông phải',
                'left_knee': 'gối trái',
                'right_knee': 'gối phải',
            }
        else:
            names = {
                'left_elbow': 'left elbow',
                'right_elbow': 'right elbow',
                'left_shoulder': 'left shoulder',
                'right_shoulder': 'right shoulder',
                'left_hip': 'left hip',
                'right_hip': 'right hip',
                'left_knee': 'left knee',
                'right_knee': 'right knee',
            }
        return names.get(joint_name, joint_name)
    
    def _generate_suggestions(
        self,
        errors: List[JointError],
        pose_name: Optional[str]
    ) -> List[str]:
        """Generate prioritized suggestions."""
        suggestions = []
        
        # Add error-based suggestions (max 3)
        for error in errors[:3]:
            suggestions.append(error.suggestion)
        
        # Add pose-specific tips
        if pose_name and pose_name in YOGA_POSES:
            pose = YOGA_POSES[pose_name]
            if self.language == 'vi':
                suggestions.append(f"Lưu ý: {pose.description}")
            else:
                suggestions.append(f"Note: {pose.description}")
        
        return suggestions
    
    def _generate_positive_feedback(self, result: ComparisonResult) -> List[str]:
        """Generate positive feedback for correct joints."""
        positive = []
        
        # Find joints with good scores
        for joint, score in result.joint_scores.items():
            if score >= 85 and joint in JOINT_ANGLES:
                joint_display = self._get_joint_display_name(joint)
                positive.append(
                    self.feedback_templates['joint_correct'].format(joint=joint_display)
                )
        
        return positive[:3]  # Max 3 positive comments
    
    def generate_realtime_feedback(
        self,
        comparison_result: ComparisonResult,
        max_messages: int = 2
    ) -> List[str]:
        """
        Generate concise real-time feedback for display during exercise.
        
        Args:
            comparison_result: Comparison result
            max_messages: Maximum number of messages
        
        Returns:
            List of short feedback messages
        """
        messages = []
        
        # Overall status
        score = comparison_result.overall_score
        if score >= 90:
            if self.language == 'vi':
                messages.append("✓ Tuyệt vời!")
            else:
                messages.append("✓ Perfect!")
        elif score >= 75:
            if self.language == 'vi':
                messages.append("○ Tốt, giữ vững!")
            else:
                messages.append("○ Good, hold it!")
        
        # Top errors
        for joint in comparison_result.error_joints[:max_messages]:
            if joint in FEEDBACK_MESSAGES:
                # Determine error type from angle difference
                if joint in comparison_result.angle_differences:
                    diff = comparison_result.angle_differences[joint]
                    error_type = 'too_bent' if diff < 0 else 'too_straight'
                    msg = FEEDBACK_MESSAGES[joint].get(error_type, '')
                    if msg:
                        messages.append(f"→ {msg}")
        
        return messages[:max_messages + 1]
    
    def generate_exercise_report(
        self,
        feedback_history: List[FeedbackResult],
        exercise_name: str
    ) -> Dict:
        """
        Generate summary report after exercise session.
        
        Args:
            feedback_history: List of feedback results from session
            exercise_name: Name of exercise performed
        
        Returns:
            Dict with session summary
        """
        import numpy as np
        
        if not feedback_history:
            return {'error': 'No feedback data'}
        
        scores = [f.overall_score for f in feedback_history]
        
        # Aggregate errors
        error_counts = {}
        for fb in feedback_history:
            for error in fb.errors:
                key = error.joint_name
                if key not in error_counts:
                    error_counts[key] = 0
                error_counts[key] += 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])
        
        # Generate summary
        avg_score = np.mean(scores)
        best_score = np.max(scores)
        worst_score = np.min(scores)
        
        if self.language == 'vi':
            summary = {
                'exercise': exercise_name,
                'average_score': avg_score,
                'best_score': best_score,
                'worst_score': worst_score,
                'total_frames': len(feedback_history),
                'common_errors': [
                    {
                        'joint': self._get_joint_display_name(j),
                        'frequency': c,
                        'percentage': c / len(feedback_history) * 100
                    }
                    for j, c in sorted_errors[:3]
                ],
                'recommendations': self._generate_recommendations(sorted_errors, exercise_name)
            }
        else:
            summary = {
                'exercise': exercise_name,
                'average_score': avg_score,
                'best_score': best_score,
                'worst_score': worst_score,
                'total_frames': len(feedback_history),
                'common_errors': [
                    {
                        'joint': self._get_joint_display_name(j),
                        'frequency': c,
                        'percentage': c / len(feedback_history) * 100
                    }
                    for j, c in sorted_errors[:3]
                ],
                'recommendations': self._generate_recommendations(sorted_errors, exercise_name)
            }
        
        return summary
    
    def _generate_recommendations(
        self,
        sorted_errors: List[Tuple[str, int]],
        exercise_name: str
    ) -> List[str]:
        """Generate recommendations based on common errors."""
        recommendations = []
        
        if not sorted_errors:
            if self.language == 'vi':
                recommendations.append("Tiếp tục duy trì phong độ tốt!")
            else:
                recommendations.append("Keep up the good work!")
            return recommendations
        
        # Top error recommendations
        for joint, count in sorted_errors[:2]:
            joint_display = self._get_joint_display_name(joint)
            if self.language == 'vi':
                recommendations.append(f"Tập trung cải thiện {joint_display} trong các buổi tập tiếp theo")
            else:
                recommendations.append(f"Focus on improving {joint_display} in future sessions")
        
        return recommendations
"""Tests for pose comparator."""

import numpy as np
import pytest
import sys
import os

# Thêm path để import được src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.pose_comparator import PoseComparator, ComparisonResult


class TestPoseComparator:
    """Test suite for PoseComparator."""
    
    @pytest.fixture
    def comparator(self):
        """Create comparator instance."""
        return PoseComparator(method='weighted')
    
    @pytest.fixture
    def sample_keypoints(self):
        """Create sample keypoints."""
        # 17 keypoints with x, y, confidence
        kpts = np.array([
            [100, 50, 0.9],   # nose
            [95, 45, 0.8],    # left_eye
            [105, 45, 0.8],   # right_eye
            [90, 50, 0.7],    # left_ear
            [110, 50, 0.7],   # right_ear
            [80, 100, 0.9],   # left_shoulder
            [120, 100, 0.9],  # right_shoulder
            [70, 150, 0.9],   # left_elbow
            [130, 150, 0.9],  # right_elbow
            [65, 200, 0.8],   # left_wrist
            [135, 200, 0.8],  # right_wrist
            [85, 200, 0.9],   # left_hip
            [115, 200, 0.9],  # right_hip
            [80, 280, 0.9],   # left_knee
            [120, 280, 0.9],  # right_knee
            [75, 350, 0.8],   # left_ankle
            [125, 350, 0.8],  # right_ankle
        ], dtype=np.float32)
        return kpts
    
    def test_compare_identical_poses(self, comparator, sample_keypoints):
        """Test comparison of identical poses."""
        result = comparator.compare(sample_keypoints, sample_keypoints)
        
        assert isinstance(result, ComparisonResult)
        assert result.overall_score >= 95.0  # Should be very high
        assert len(result.error_joints) == 0
    
    def test_compare_different_poses(self, comparator, sample_keypoints):
        """Test comparison of different poses."""
        modified = sample_keypoints.copy()
        modified[7, :2] += 50  # Move left elbow significantly
        modified[8, :2] += 50  # Move right elbow
        
        result = comparator.compare(sample_keypoints, modified)
        
        assert result.overall_score < 90.0
        assert 'left_elbow' in result.error_joints or 'right_elbow' in result.error_joints
    
    def test_cosine_method(self, comparator, sample_keypoints):
        """Test cosine similarity method."""
        result = comparator.compare(sample_keypoints, sample_keypoints, method='cosine')
        
        assert result.method == 'cosine'
        assert result.overall_score >= 95.0
    
    def test_angle_method(self, comparator, sample_keypoints):
        """Test angle comparison method."""
        result = comparator.compare(sample_keypoints, sample_keypoints, method='angle')
        
        assert result.method == 'angle'
        assert 'left_elbow' in result.joint_scores or 'right_elbow' in result.joint_scores
    
    def test_oks_method(self, comparator, sample_keypoints):
        """Test OKS method."""
        result = comparator.compare(sample_keypoints, sample_keypoints, method='oks')
        
        assert result.method == 'oks'
        assert result.overall_score >= 90.0
    
    def test_joint_scores_present(self, comparator, sample_keypoints):
        """Test that joint scores are calculated."""
        result = comparator.compare(sample_keypoints, sample_keypoints)
        
        assert len(result.joint_scores) > 0
        for score in result.joint_scores.values():
            assert 0 <= score <= 100


class TestMathUtils:
    """Test mathematical utilities."""
    
    def test_calculate_angle(self):
        """Test angle calculation."""
        from src.utils.math_utils import calculate_angle
        
        # 90 degree angle
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([1, 1])
        
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 1.0
    
    def test_cosine_similarity(self):
        """Test cosine similarity."""
        from src.utils.math_utils import cosine_similarity
        
        # Identical vectors
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2, 3])
        
        sim = cosine_similarity(v1, v2)
        assert abs(sim - 1.0) < 0.01
        
        # Orthogonal vectors
        v3 = np.array([1, 0])
        v4 = np.array([0, 1])
        
        sim_ortho = cosine_similarity(v3, v4)
        assert abs(sim_ortho) < 0.01
    
    def test_normalize_keypoints(self):
        """Test keypoint normalization."""
        from src.utils.math_utils import normalize_keypoints
        
        kpts = np.array([
            [100, 100],
            [200, 100],
            [150, 200],
        ], dtype=np.float32)
        
        normalized = normalize_keypoints(kpts, method='minmax')
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestFeedbackGenerator:
    """Test feedback generation."""
    
    def test_rating_calculation(self):
        """Test rating based on score."""
        from src.core.feedback_generator import FeedbackGenerator
        
        gen = FeedbackGenerator(language='en')
        
        assert gen._get_rating(95) == 'excellent'
        assert gen._get_rating(80) == 'good'
        assert gen._get_rating(65) == 'needs_improvement'
        assert gen._get_rating(50) == 'poor'
    
    def test_vietnamese_language(self):
        """Test Vietnamese feedback."""
        from src.core.feedback_generator import FeedbackGenerator
        
        gen_vi = FeedbackGenerator(language='vi')
        gen_en = FeedbackGenerator(language='en')
        
        assert gen_vi._get_joint_display_name('left_knee') == 'gối trái'
        assert gen_en._get_joint_display_name('left_knee') == 'left knee'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
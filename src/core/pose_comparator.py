"""Pose comparison algorithms."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..config.pose_config import (
    JOINT_ANGLES,
    KEYPOINT_WEIGHTS,
    OKS_SIGMAS,
    ERROR_THRESHOLDS,
)
from ..utils.math_utils import (
    normalize_keypoints,
    cosine_similarity,
    weighted_cosine_similarity,
    oks_similarity,
    keypoints_to_angles,
    angle_difference,
    calculate_body_bbox,
)


@dataclass
class ComparisonResult:
    """Result of pose comparison."""
    overall_score: float
    method: str
    joint_scores: Dict[str, float] = field(default_factory=dict)
    angle_differences: Dict[str, float] = field(default_factory=dict)
    error_joints: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


class PoseComparator:
    """
    Compare two poses using multiple methods.
    
    Methods:
        - cosine: Fast, scale-invariant vector similarity
        - angle: Compare joint angles (more intuitive for feedback)
        - oks: Object Keypoint Similarity (COCO standard)
        - weighted: Weighted combination of above
    """
    
    def __init__(
        self,
        method: str = 'weighted',
        angle_tolerance: float = None,
        confidence_threshold: float = None
    ):
        """
        Initialize pose comparator.
        
        Args:
            method: Comparison method ('cosine', 'angle', 'oks', 'weighted')
            angle_tolerance: Tolerance for angle comparison (degrees)
            confidence_threshold: Minimum keypoint confidence
        """
        self.method = method
        self.angle_tolerance = angle_tolerance or ERROR_THRESHOLDS['angle_tolerance']
        self.confidence_threshold = confidence_threshold or ERROR_THRESHOLDS['confidence_threshold']
        
        # Weights for different joints (from config)
        self.keypoint_weights = np.array([
            KEYPOINT_WEIGHTS[kp] for kp in [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
        ])
        
        self.oks_sigmas = np.array(OKS_SIGMAS)
    
    def compare(
        self,
        keypoints_user: np.ndarray,
        keypoints_ref: np.ndarray,
        method: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare two poses.
        
        Args:
            keypoints_user: User's keypoints (17, 2) or (17, 3)
            keypoints_ref: Reference keypoints (17, 2) or (17, 3)
            method: Override default method
        
        Returns:
            ComparisonResult with scores and details
        """
        method = method or self.method
        
        if method == 'cosine':
            return self._compare_cosine(keypoints_user, keypoints_ref)
        elif method == 'angle':
            return self._compare_angles(keypoints_user, keypoints_ref)
        elif method == 'oks':
            return self._compare_oks(keypoints_user, keypoints_ref)
        elif method == 'weighted':
            return self._compare_weighted(keypoints_user, keypoints_ref)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compare_cosine(
        self,
        kpts_user: np.ndarray,
        kpts_ref: np.ndarray
    ) -> ComparisonResult:
        """Compare using cosine similarity."""
        # Normalize keypoints
        norm_user = normalize_keypoints(kpts_user, method='torso')
        norm_ref = normalize_keypoints(kpts_ref, method='torso')
        
        # Flatten to vectors
        vec_user = norm_user[:, :2].flatten()
        vec_ref = norm_ref[:, :2].flatten()
        
        # Calculate similarity
        similarity = cosine_similarity(vec_user, vec_ref)
        score = (similarity + 1) / 2 * 100  # Convert to 0-100 scale
        
        # Calculate per-joint scores
        joint_scores = {}
        coords_user = norm_user[:, :2]
        coords_ref = norm_ref[:, :2]
        
        joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for i, name in enumerate(joint_names):
            dist = np.linalg.norm(coords_user[i] - coords_ref[i])
            joint_scores[name] = max(0, 100 - dist * 100)
        
        # Find error joints
        error_joints = [name for name, s in joint_scores.items() if s < 70]
        
        return ComparisonResult(
            overall_score=score,
            method='cosine',
            joint_scores=joint_scores,
            error_joints=error_joints,
            details={'raw_similarity': similarity}
        )
    
    def _compare_angles(
        self,
        kpts_user: np.ndarray,
        kpts_ref: np.ndarray
    ) -> ComparisonResult:
        """Compare using joint angles."""
        # Calculate angles for both poses
        angles_user = keypoints_to_angles(kpts_user, JOINT_ANGLES)
        angles_ref = keypoints_to_angles(kpts_ref, JOINT_ANGLES)
        
        # Calculate differences
        angle_diffs = {}
        joint_scores = {}
        error_joints = []
        
        for joint_name in JOINT_ANGLES.keys():
            angle_u = angles_user.get(joint_name)
            angle_r = angles_ref.get(joint_name)
            
            if angle_u is None or angle_r is None:
                continue
            
            diff = angle_difference(angle_u, angle_r)
            angle_diffs[joint_name] = diff
            
            # Convert to score (0 diff = 100%, tolerance diff = 70%)
            score = max(0, 100 - (diff / self.angle_tolerance) * 30)
            joint_scores[joint_name] = score
            
            if diff > self.angle_tolerance:
                error_joints.append(joint_name)
        
        # Overall score is weighted average
        if joint_scores:
            weights = [KEYPOINT_WEIGHTS.get(j.split('_')[0] + '_' + j.split('_')[1], 1.0) 
                      for j in joint_scores.keys()]
            overall_score = np.average(list(joint_scores.values()), weights=weights)
        else:
            overall_score = 0.0
        
        return ComparisonResult(
            overall_score=overall_score,
            method='angle',
            joint_scores=joint_scores,
            angle_differences=angle_diffs,
            error_joints=error_joints,
            details={'angles_user': angles_user, 'angles_ref': angles_ref}
        )
    
    def _compare_oks(
        self,
        kpts_user: np.ndarray,
        kpts_ref: np.ndarray
    ) -> ComparisonResult:
        """Compare using Object Keypoint Similarity."""
        # Get bounding box scale
        bbox = calculate_body_bbox(kpts_ref)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        scale = np.sqrt(area) if area > 0 else 1.0
        
        # Calculate OKS
        oks = oks_similarity(kpts_user, kpts_ref, self.oks_sigmas, scale)
        score = oks * 100
        
        # Per-keypoint OKS
        joint_scores = {}
        coords_user = kpts_user[:, :2] if kpts_user.shape[1] == 3 else kpts_user
        coords_ref = kpts_ref[:, :2] if kpts_ref.shape[1] == 3 else kpts_ref
        
        joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for i, name in enumerate(joint_names):
            d_sq = np.sum((coords_user[i] - coords_ref[i]) ** 2)
            var = (self.oks_sigmas[i] * 2) ** 2
            kpt_oks = np.exp(-d_sq / (2 * scale * var + 1e-8))
            joint_scores[name] = kpt_oks * 100
        
        error_joints = [name for name, s in joint_scores.items() if s < 70]
        
        return ComparisonResult(
            overall_score=score,
            method='oks',
            joint_scores=joint_scores,
            error_joints=error_joints,
            details={'raw_oks': oks, 'scale': scale}
        )
    
    def _compare_weighted(
        self,
        kpts_user: np.ndarray,
        kpts_ref: np.ndarray
    ) -> ComparisonResult:
        """Weighted combination of multiple methods."""
        # Get results from each method
        cosine_result = self._compare_cosine(kpts_user, kpts_ref)
        angle_result = self._compare_angles(kpts_user, kpts_ref)
        oks_result = self._compare_oks(kpts_user, kpts_ref)
        
        # Weighted combination
        weights = {'cosine': 0.2, 'angle': 0.5, 'oks': 0.3}
        overall_score = (
            cosine_result.overall_score * weights['cosine'] +
            angle_result.overall_score * weights['angle'] +
            oks_result.overall_score * weights['oks']
        )
        
        # Combine joint scores
        joint_scores = {}
        all_joints = set(cosine_result.joint_scores.keys()) | set(angle_result.joint_scores.keys())
        
        for joint in all_joints:
            scores = []
            if joint in cosine_result.joint_scores:
                scores.append(cosine_result.joint_scores[joint])
            if joint in angle_result.joint_scores:
                scores.append(angle_result.joint_scores[joint])
            if joint in oks_result.joint_scores:
                scores.append(oks_result.joint_scores[joint])
            joint_scores[joint] = np.mean(scores) if scores else 0
        
        # Combine error joints
        error_joints = list(set(
            angle_result.error_joints +  # Prioritize angle errors
            [j for j in oks_result.error_joints if j not in angle_result.error_joints]
        ))
        
        return ComparisonResult(
            overall_score=overall_score,
            method='weighted',
            joint_scores=joint_scores,
            angle_differences=angle_result.angle_differences,
            error_joints=error_joints,
            details={
                'cosine_score': cosine_result.overall_score,
                'angle_score': angle_result.overall_score,
                'oks_score': oks_result.overall_score,
                'weights': weights
            }
        )
    
    def compare_sequences(
        self,
        sequence_user: List[np.ndarray],
        sequence_ref: List[np.ndarray],
        use_dtw: bool = True
    ) -> Dict:
        """
        Compare two pose sequences (for video comparison).
        
        Args:
            sequence_user: List of user keypoints per frame
            sequence_ref: List of reference keypoints per frame
            use_dtw: Use Dynamic Time Warping for alignment
        
        Returns:
            Dict with sequence comparison results
        """
        if use_dtw:
            return self._compare_with_dtw(sequence_user, sequence_ref)
        else:
            return self._compare_frame_by_frame(sequence_user, sequence_ref)
    
    def _compare_frame_by_frame(
        self,
        seq_user: List[np.ndarray],
        seq_ref: List[np.ndarray]
    ) -> Dict:
        """Simple frame-by-frame comparison."""
        # Match lengths
        min_len = min(len(seq_user), len(seq_ref))
        
        frame_scores = []
        frame_errors = []
        
        for i in range(min_len):
            result = self.compare(seq_user[i], seq_ref[i])
            frame_scores.append(result.overall_score)
            frame_errors.append(result.error_joints)
        
        return {
            'overall_score': np.mean(frame_scores),
            'frame_scores': frame_scores,
            'frame_errors': frame_errors,
            'best_frame': int(np.argmax(frame_scores)),
            'worst_frame': int(np.argmin(frame_scores)),
            'method': 'frame_by_frame'
        }
    
    def _compare_with_dtw(
        self,
        seq_user: List[np.ndarray],
        seq_ref: List[np.ndarray]
    ) -> Dict:
        """Compare using Dynamic Time Warping."""
        try:
            from dtaidistance import dtw
        except ImportError:
            # Fallback to simple comparison
            return self._compare_frame_by_frame(seq_user, seq_ref)
        
        # Convert to angle sequences for each joint
        joint_dtw_scores = {}
        
        for joint_name, indices in JOINT_ANGLES.items():
            # Extract angle sequence for this joint
            angles_user = []
            angles_ref = []
            
            for kpts in seq_user:
                angles = keypoints_to_angles(kpts, {joint_name: indices})
                if angles[joint_name] is not None:
                    angles_user.append(angles[joint_name])
            
            for kpts in seq_ref:
                angles = keypoints_to_angles(kpts, {joint_name: indices})
                if angles[joint_name] is not None:
                    angles_ref.append(angles[joint_name])
            
            if len(angles_user) < 2 or len(angles_ref) < 2:
                continue
            
            # Calculate DTW distance
            distance = dtw.distance(
                np.array(angles_user),
                np.array(angles_ref)
            )
            
            # Normalize by sequence length
            max_len = max(len(angles_user), len(angles_ref))
            normalized_dist = distance / max_len
            
            # Convert to score (lower distance = higher score)
            score = max(0, 100 - normalized_dist * 2)
            joint_dtw_scores[joint_name] = score
        
        # Overall score
        if joint_dtw_scores:
            overall_score = np.mean(list(joint_dtw_scores.values()))
        else:
            overall_score = 0
        
        # Find error joints
        error_joints = [j for j, s in joint_dtw_scores.items() if s < 70]
        
        return {
            'overall_score': overall_score,
            'joint_scores': joint_dtw_scores,
            'error_joints': error_joints,
            'method': 'dtw'
        }
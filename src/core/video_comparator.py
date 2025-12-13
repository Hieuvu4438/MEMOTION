"""Video-to-video pose comparison."""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json

from ..utils.video_utils import VideoReader, VideoWriter, get_video_info
from ..utils.visualization import draw_skeleton, draw_keypoints, draw_feedback_overlay
from .pose_assessor import PoseAssessor
from .pose_comparator import PoseComparator


@dataclass
class VideoComparisonResult:
    """Result of video comparison."""
    overall_score: float
    frame_scores: List[float]
    joint_scores: Dict[str, float]
    error_joints: List[str]
    best_frame_idx: int
    worst_frame_idx: int
    method: str
    alignment_path: Optional[List[Tuple[int, int]]] = None
    details: Dict = field(default_factory=dict)


class VideoComparator:
    """Compare two videos using pose estimation."""
    
    def __init__(
        self,
        model_size: str = 's',
        device: str = 'cpu',
        target_fps: float = 15.0
    ):
        """
        Initialize video comparator.
        
        Args:
            model_size: RTMPose model size
            device: 'cpu' or 'cuda'
            target_fps: Target FPS for processing
        """
        self.assessor = PoseAssessor(model_size=model_size, device=device)
        self.comparator = PoseComparator(method='weighted')
        self.target_fps = target_fps
    
    def compare(
        self,
        user_video: str,
        reference_video: str,
        use_dtw: bool = True,
        output_video: Optional[str] = None
    ) -> VideoComparisonResult:
        """
        Compare user video with reference video.
        
        Args:
            user_video: Path to user's video
            reference_video: Path to reference video
            use_dtw: Use Dynamic Time Warping for temporal alignment
            output_video: Optional path to save comparison video
        
        Returns:
            VideoComparisonResult
        """
        # Extract poses from both videos
        user_poses = self._extract_poses(user_video)
        ref_poses = self._extract_poses(reference_video)
        
        if use_dtw:
            result = self._compare_with_dtw(user_poses, ref_poses)
        else:
            result = self._compare_frame_by_frame(user_poses, ref_poses)
        
        # Generate output video if requested
        if output_video:
            self._generate_comparison_video(
                user_video, reference_video,
                user_poses, ref_poses,
                result, output_video
            )
        
        return result
    
    def _extract_poses(self, video_path: str) -> List[np.ndarray]:
        """Extract pose keypoints from video."""
        poses = []
        
        with VideoReader(video_path, target_fps=self.target_fps) as reader:
            for _, frame in reader:
                keypoints = self.assessor.estimate_pose(frame)
                poses.append(keypoints)
        
        return poses
    
    def _compare_frame_by_frame(
        self,
        user_poses: List[np.ndarray],
        ref_poses: List[np.ndarray]
    ) -> VideoComparisonResult:
        """Simple frame-by-frame comparison."""
        # Match lengths by interpolation
        if len(user_poses) != len(ref_poses):
            user_poses, ref_poses = self._match_lengths(user_poses, ref_poses)
        
        frame_scores = []
        all_joint_scores = {}
        all_errors = []
        
        for user_kpts, ref_kpts in zip(user_poses, ref_poses):
            result = self.comparator.compare(user_kpts, ref_kpts)
            frame_scores.append(result.overall_score)
            all_errors.extend(result.error_joints)
            
            for joint, score in result.joint_scores.items():
                if joint not in all_joint_scores:
                    all_joint_scores[joint] = []
                all_joint_scores[joint].append(score)
        
        # Aggregate results
        joint_scores = {j: np.mean(s) for j, s in all_joint_scores.items()}
        
        # Count error frequency
        error_counts = {}
        for e in all_errors:
            error_counts[e] = error_counts.get(e, 0) + 1
        
        # Top errors (appear in >30% of frames)
        threshold = len(frame_scores) * 0.3
        error_joints = [e for e, c in error_counts.items() if c > threshold]
        
        return VideoComparisonResult(
            overall_score=np.mean(frame_scores),
            frame_scores=frame_scores,
            joint_scores=joint_scores,
            error_joints=error_joints,
            best_frame_idx=int(np.argmax(frame_scores)),
            worst_frame_idx=int(np.argmin(frame_scores)),
            method='frame_by_frame',
            details={'error_counts': error_counts}
        )
    
    def _compare_with_dtw(
        self,
        user_poses: List[np.ndarray],
        ref_poses: List[np.ndarray]
    ) -> VideoComparisonResult:
        """Compare using Dynamic Time Warping."""
        try:
            from dtaidistance import dtw
            from dtaidistance.dtw import warping_path
        except ImportError:
            return self._compare_frame_by_frame(user_poses, ref_poses)
        
        # Flatten poses to feature vectors
        user_features = np.array([p[:, :2].flatten() for p in user_poses])
        ref_features = np.array([p[:, :2].flatten() for p in ref_poses])
        
        # Normalize
        user_norm = (user_features - user_features.mean(axis=0)) / (user_features.std(axis=0) + 1e-8)
        ref_norm = (ref_features - ref_features.mean(axis=0)) / (ref_features.std(axis=0) + 1e-8)
        
        # Calculate DTW path
        path = warping_path(user_norm.flatten(), ref_norm.flatten())
        
        # Calculate aligned frame scores
        frame_scores = []
        aligned_pairs = self._dtw_to_frame_pairs(path, len(user_poses), len(ref_poses))
        
        all_joint_scores = {}
        all_errors = []
        
        for user_idx, ref_idx in aligned_pairs:
            result = self.comparator.compare(user_poses[user_idx], ref_poses[ref_idx])
            frame_scores.append(result.overall_score)
            all_errors.extend(result.error_joints)
            
            for joint, score in result.joint_scores.items():
                if joint not in all_joint_scores:
                    all_joint_scores[joint] = []
                all_joint_scores[joint].append(score)
        
        joint_scores = {j: np.mean(s) for j, s in all_joint_scores.items()}
        
        # Error analysis
        error_counts = {}
        for e in all_errors:
            error_counts[e] = error_counts.get(e, 0) + 1
        
        threshold = len(frame_scores) * 0.3
        error_joints = [e for e, c in error_counts.items() if c > threshold]
        
        return VideoComparisonResult(
            overall_score=np.mean(frame_scores),
            frame_scores=frame_scores,
            joint_scores=joint_scores,
            error_joints=error_joints,
            best_frame_idx=int(np.argmax(frame_scores)),
            worst_frame_idx=int(np.argmin(frame_scores)),
            method='dtw',
            alignment_path=aligned_pairs,
            details={'error_counts': error_counts}
        )
    
    def _dtw_to_frame_pairs(
        self,
        dtw_path: List[Tuple[int, int]],
        user_len: int,
        ref_len: int
    ) -> List[Tuple[int, int]]:
        """Convert DTW path to frame pairs."""
        # Sample path to get reasonable number of pairs
        pairs = []
        
        # DTW path is for flattened features, convert back to frames
        user_step = user_len * 34  # 17 keypoints * 2 coords
        ref_step = ref_len * 34
        
        visited_user = set()
        visited_ref = set()
        
        for i, (u, r) in enumerate(dtw_path[::100]):  # Sample
            user_frame = min(u // 34, user_len - 1)
            ref_frame = min(r // 34, ref_len - 1)
            
            if user_frame not in visited_user or ref_frame not in visited_ref:
                pairs.append((user_frame, ref_frame))
                visited_user.add(user_frame)
                visited_ref.add(ref_frame)
        
        if not pairs:
            # Fallback to simple matching
            for i in range(min(user_len, ref_len)):
                pairs.append((i, i))
        
        return pairs
    
    def _match_lengths(
        self,
        poses1: List[np.ndarray],
        poses2: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Match lengths by interpolation."""
        len1, len2 = len(poses1), len(poses2)
        target_len = max(len1, len2)
        
        if len1 < target_len:
            indices = np.linspace(0, len1 - 1, target_len).astype(int)
            poses1 = [poses1[i] for i in indices]
        
        if len2 < target_len:
            indices = np.linspace(0, len2 - 1, target_len).astype(int)
            poses2 = [poses2[i] for i in indices]
        
        return poses1, poses2
    
    def _generate_comparison_video(
        self,
        user_video: str,
        ref_video: str,
        user_poses: List[np.ndarray],
        ref_poses: List[np.ndarray],
        result: VideoComparisonResult,
        output_path: str
    ):
        """Generate side-by-side comparison video."""
        user_info = get_video_info(user_video)
        ref_info = get_video_info(ref_video)
        
        # Target dimensions
        target_h = 480
        target_w = 640
        
        with VideoReader(user_video, target_fps=self.target_fps) as user_reader, \
             VideoReader(ref_video, target_fps=self.target_fps) as ref_reader, \
             VideoWriter(output_path, self.target_fps, (target_w * 2 + 20, target_h)) as writer:
            
            user_frames = list(user_reader)
            ref_frames = list(ref_reader)
            
            # Match frame counts
            min_frames = min(len(user_frames), len(ref_frames), len(result.frame_scores))
            
            for i in range(min_frames):
                _, user_frame = user_frames[i]
                _, ref_frame = ref_frames[i]
                
                # Resize
                user_frame = cv2.resize(user_frame, (target_w, target_h))
                ref_frame = cv2.resize(ref_frame, (target_w, target_h))
                
                # Draw poses
                if i < len(user_poses):
                    # Scale keypoints
                    user_kpts = user_poses[i].copy()
                    user_kpts[:, 0] *= target_w / user_info['width']
                    user_kpts[:, 1] *= target_h / user_info['height']
                    user_frame = draw_skeleton(user_frame, user_kpts)
                
                if i < len(ref_poses):
                    ref_kpts = ref_poses[i].copy()
                    ref_kpts[:, 0] *= target_w / ref_info['width']
                    ref_kpts[:, 1] *= target_h / ref_info['height']
                    ref_frame = draw_skeleton(ref_frame, ref_kpts)
                
                # Add labels
                cv2.putText(user_frame, "User", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(ref_frame, "Reference", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add score
                score = result.frame_scores[i] if i < len(result.frame_scores) else 0
                color = (0, 255, 0) if score >= 80 else ((0, 165, 255) if score >= 60 else (0, 0, 255))
                cv2.putText(user_frame, f"Score: {score:.1f}%", (10, target_h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Combine
                separator = np.ones((target_h, 20, 3), dtype=np.uint8) * 50
                combined = np.hstack([user_frame, separator, ref_frame])
                
                writer.write(combined)
    
    def generate_report(
        self,
        result: VideoComparisonResult,
        output_path: str,
        format: str = 'html'
    ):
        """
        Generate assessment report.
        
        Args:
            result: VideoComparisonResult
            output_path: Output file path
            format: 'html' or 'json'
        """
        if format == 'json':
            self._generate_json_report(result, output_path)
        else:
            self._generate_html_report(result, output_path)
    
    def _generate_json_report(self, result: VideoComparisonResult, output_path: str):
        """Generate JSON report."""
        report = {
            'overall_score': result.overall_score,
            'method': result.method,
            'frame_scores': result.frame_scores,
            'joint_scores': result.joint_scores,
            'error_joints': result.error_joints,
            'best_frame': result.best_frame_idx,
            'worst_frame': result.worst_frame_idx,
            'statistics': {
                'mean': np.mean(result.frame_scores),
                'std': np.std(result.frame_scores),
                'min': np.min(result.frame_scores),
                'max': np.max(result.frame_scores)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_html_report(self, result: VideoComparisonResult, output_path: str):
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pose Assessment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .score {{ font-size: 48px; font-weight: bold; }}
        .good {{ color: green; }}
        .moderate {{ color: orange; }}
        .poor {{ color: red; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        .bar {{ height: 20px; background: #4CAF50; }}
    </style>
</head>
<body>
    <h1>Pose Assessment Report</h1>
    
    <div class="section">
        <h2>Overall Score</h2>
        <div class="score {'good' if result.overall_score >= 80 else 'moderate' if result.overall_score >= 60 else 'poor'}">
            {result.overall_score:.1f}%
        </div>
    </div>
    
    <div class="section">
        <h2>Joint Analysis</h2>
        <table>
            <tr><th>Joint</th><th>Score</th><th>Visual</th></tr>
            {''.join(f'<tr><td>{j}</td><td>{s:.1f}%</td><td><div class="bar" style="width:{s}%"></div></td></tr>' for j, s in sorted(result.joint_scores.items(), key=lambda x: -x[1]))}
        </table>
    </div>
    
    <div class="section">
        <h2>Areas for Improvement</h2>
        <ul>
            {''.join(f'<li>{joint}</li>' for joint in result.error_joints)}
        </ul>
    </div>
    
    <div class="section">
        <h2>Statistics</h2>
        <p>Best frame: #{result.best_frame_idx} ({max(result.frame_scores):.1f}%)</p>
        <p>Worst frame: #{result.worst_frame_idx} ({min(result.frame_scores):.1f}%)</p>
        <p>Average: {np.mean(result.frame_scores):.1f}%</p>
        <p>Std deviation: {np.std(result.frame_scores):.1f}%</p>
    </div>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html)
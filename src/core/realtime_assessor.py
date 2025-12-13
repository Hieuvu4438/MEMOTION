"""Real-time pose assessment from camera."""

import cv2
import numpy as np
import time
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..utils.video_utils import VideoReader, VideoWriter
from ..utils.visualization import (
    draw_skeleton,
    draw_keypoints,
    draw_feedback_overlay,
    create_side_by_side_comparison,
    COLORS
)
from .pose_assessor import PoseAssessor


@dataclass
class RealtimeConfig:
    """Configuration for real-time assessment."""
    show_skeleton: bool = True
    show_reference: bool = True
    show_feedback: bool = True
    show_fps: bool = True
    mirror: bool = True
    target_fps: float = 30.0
    feedback_update_interval: float = 0.5  # seconds
    window_name: str = "Pose Assessment"


class RealtimeAssessor:
    """Real-time pose assessment from camera with visual feedback."""
    
    def __init__(
        self,
        reference_source: Optional[str] = None,
        model_size: str = 's',
        device: str = 'cpu',
        language: str = 'vi',
        config: Optional[RealtimeConfig] = None
    ):
        """
        Initialize real-time assessor.
        
        Args:
            reference_source: Path to reference image or video
            model_size: RTMPose model size
            device: 'cpu' or 'cuda'
            language: Feedback language
            config: RealtimeConfig settings
        """
        self.assessor = PoseAssessor(
            model_size=model_size,
            device=device,
            language=language
        )
        
        self.config = config or RealtimeConfig()
        self.reference_keypoints = None
        self.reference_image = None
        
        if reference_source:
            self.load_reference(reference_source)
        
        # State
        self._running = False
        self._last_feedback_time = 0
        self._current_feedback = []
        self._current_score = 0
        self._fps_history = []
    
    def load_reference(self, source: str):
        """
        Load reference pose from image or video.
        
        Args:
            source: Path to image or video file
        """
        path = Path(source)
        
        if path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Extract middle frame from video
            with VideoReader(source) as reader:
                mid_frame = reader.frame_count // 2
                frame = reader.get_frame(mid_frame)
                if frame is not None:
                    self.reference_image = frame
                    self.reference_keypoints = self.assessor.estimate_pose(frame)
        else:
            # Load image
            self.reference_image = cv2.imread(source)
            if self.reference_image is not None:
                self.reference_keypoints = self.assessor.estimate_pose(self.reference_image)
    
    def start(
        self,
        camera_id: int = 0,
        output_path: Optional[str] = None,
        callback: Optional[Callable] = None
    ):
        """
        Start real-time assessment.
        
        Args:
            camera_id: Camera device ID
            output_path: Optional path to save video
            callback: Optional callback function(score, feedback, frame)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
        
        self._running = True
        
        try:
            while self._running:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror if configured
                if self.config.mirror:
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                output_frame = self._process_frame(frame)
                
                # FPS calculation
                fps = 1.0 / (time.time() - start_time + 1e-8)
                self._fps_history.append(fps)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)
                
                # Draw FPS
                if self.config.show_fps:
                    avg_fps = np.mean(self._fps_history)
                    cv2.putText(
                        output_frame, f"FPS: {avg_fps:.1f}",
                        (10, output_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1
                    )
                
                # Callback
                if callback:
                    callback(self._current_score, self._current_feedback, output_frame)
                
                # Save to video
                if writer:
                    writer.write(output_frame)
                
                # Display
                cv2.imshow(self.config.window_name, output_frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('r'):  # Reset reference
                    self._capture_reference(frame)
                elif key == ord('s'):  # Screenshot
                    self._save_screenshot(output_frame)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self._running = False
    
    def stop(self):
        """Stop real-time assessment."""
        self._running = False
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame and return visualization."""
        output = frame.copy()
        
        # Estimate user pose
        user_kpts = self.assessor.estimate_pose(frame)
        
        # Draw skeleton
        if self.config.show_skeleton:
            output = draw_skeleton(output, user_kpts)
            output = draw_keypoints(output, user_kpts)
        
        # Compare with reference
        if self.reference_keypoints is not None:
            current_time = time.time()
            
            # Update feedback at intervals (not every frame)
            if current_time - self._last_feedback_time > self.config.feedback_update_interval:
                score, messages, _ = self.assessor.get_realtime_feedback(
                    frame, self.reference_keypoints
                )
                self._current_score = score
                self._current_feedback = messages
                self._last_feedback_time = current_time
            
            # Highlight errors
            comparison = self.assessor.compare_poses(user_kpts, self.reference_keypoints)
            for joint in comparison.error_joints:
                from ..config.pose_config import KEYPOINT_INDEX
                if joint in KEYPOINT_INDEX:
                    idx = KEYPOINT_INDEX[joint]
                    pt = user_kpts[idx, :2].astype(int)
                    if pt[0] > 0 and pt[1] > 0:
                        cv2.circle(output, tuple(pt), 15, COLORS['error'], 3)
            
            # Draw feedback overlay
            if self.config.show_feedback:
                output = draw_feedback_overlay(
                    output, self._current_feedback, self._current_score
                )
        
        # Show reference in corner
        if self.config.show_reference and self.reference_image is not None:
            output = self._add_reference_thumbnail(output)
        
        return output
    
    def _add_reference_thumbnail(self, frame: np.ndarray) -> np.ndarray:
        """Add reference image thumbnail to corner."""
        h, w = frame.shape[:2]
        thumb_h, thumb_w = h // 4, w // 4
        
        # Resize reference
        thumb = cv2.resize(self.reference_image, (thumb_w, thumb_h))
        
        # Draw skeleton on thumbnail
        if self.reference_keypoints is not None:
            scale_x = thumb_w / self.reference_image.shape[1]
            scale_y = thumb_h / self.reference_image.shape[0]
            scaled_kpts = self.reference_keypoints.copy()
            scaled_kpts[:, 0] *= scale_x
            scaled_kpts[:, 1] *= scale_y
            thumb = draw_skeleton(thumb, scaled_kpts)
        
        # Add border
        cv2.rectangle(thumb, (0, 0), (thumb_w-1, thumb_h-1), (255, 255, 255), 2)
        
        # Add label
        cv2.putText(thumb, "Reference", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Place in corner
        frame[10:10+thumb_h, w-thumb_w-10:w-10] = thumb
        
        return frame
    
    def _capture_reference(self, frame: np.ndarray):
        """Capture current frame as new reference."""
        self.reference_image = frame.copy()
        self.reference_keypoints = self.assessor.estimate_pose(frame)
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save screenshot."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)


class GuidedExerciseSession:
    """Guided exercise session with multiple poses."""
    
    def __init__(
        self,
        assessor: PoseAssessor,
        poses: List[Tuple[str, str]],  # List of (name, reference_path)
        hold_time: float = 5.0,
        transition_time: float = 3.0
    ):
        """
        Initialize guided session.
        
        Args:
            assessor: PoseAssessor instance
            poses: List of (pose_name, reference_path) tuples
            hold_time: Seconds to hold each pose
            transition_time: Seconds between poses
        """
        self.assessor = assessor
        self.poses = poses
        self.hold_time = hold_time
        self.transition_time = transition_time
        
        # Pre-load references
        self.references = {}
        for name, path in poses:
            ref_img = cv2.imread(path)
            if ref_img is not None:
                self.references[name] = {
                    'image': ref_img,
                    'keypoints': assessor.estimate_pose(ref_img)
                }
        
        # Session state
        self.current_pose_idx = 0
        self.pose_start_time = 0
        self.session_scores = {}
        self.is_transitioning = False
    
    def run(self, camera_id: int = 0):
        """Run guided session."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        self.current_pose_idx = 0
        self.pose_start_time = time.time()
        
        while self.current_pose_idx < len(self.poses):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            current_time = time.time()
            elapsed = current_time - self.pose_start_time
            
            pose_name = self.poses[self.current_pose_idx][0]
            ref_data = self.references.get(pose_name)
            
            if self.is_transitioning:
                # Show transition screen
                output = self._draw_transition(frame, pose_name, elapsed)
                
                if elapsed > self.transition_time:
                    self.is_transitioning = False
                    self.pose_start_time = time.time()
            else:
                # Active pose assessment
                output = self._assess_pose(frame, pose_name, ref_data, elapsed)
                
                if elapsed > self.hold_time:
                    # Move to next pose
                    self.current_pose_idx += 1
                    self.is_transitioning = True
                    self.pose_start_time = time.time()
            
            cv2.imshow("Guided Exercise", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Return session summary
        return self._generate_summary()
    
    def _assess_pose(
        self,
        frame: np.ndarray,
        pose_name: str,
        ref_data: dict,
        elapsed: float
    ) -> np.ndarray:
        """Assess current pose and draw visualization."""
        output = frame.copy()
        
        if ref_data is None:
            return output
        
        # Get feedback
        score, messages, user_kpts = self.assessor.get_realtime_feedback(
            frame, ref_data['keypoints']
        )
        
        # Track scores
        if pose_name not in self.session_scores:
            self.session_scores[pose_name] = []
        self.session_scores[pose_name].append(score)
        
        # Draw skeleton
        output = draw_skeleton(output, user_kpts)
        
        # Draw progress bar
        progress = elapsed / self.hold_time
        bar_width = int(output.shape[1] * 0.6)
        bar_x = (output.shape[1] - bar_width) // 2
        bar_y = output.shape[0] - 40
        
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), COLORS['correct'], -1)
        
        # Draw pose name and time
        remaining = max(0, self.hold_time - elapsed)
        cv2.putText(output, f"{pose_name} - {remaining:.1f}s", (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
        
        # Draw feedback
        output = draw_feedback_overlay(output, messages, score, position='top')
        
        return output
    
    def _draw_transition(
        self,
        frame: np.ndarray,
        next_pose: str,
        elapsed: float
    ) -> np.ndarray:
        """Draw transition screen."""
        output = frame.copy()
        
        # Darken background
        overlay = np.zeros_like(output)
        output = cv2.addWeighted(output, 0.3, overlay, 0.7, 0)
        
        # Show next pose
        remaining = max(0, self.transition_time - elapsed)
        
        text = f"Next: {next_pose}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        text_x = (output.shape[1] - text_size[0]) // 2
        text_y = output.shape[0] // 2
        
        cv2.putText(output, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['text'], 2)
        
        cv2.putText(output, f"Starting in {remaining:.1f}s", (text_x, text_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['warning'], 2)
        
        # Show reference thumbnail
        ref_data = self.references.get(next_pose)
        if ref_data is not None:
            thumb_h = output.shape[0] // 3
            thumb_w = int(thumb_h * ref_data['image'].shape[1] / ref_data['image'].shape[0])
            thumb = cv2.resize(ref_data['image'], (thumb_w, thumb_h))
            
            x = (output.shape[1] - thumb_w) // 2
            y = text_y + 80
            
            if y + thumb_h < output.shape[0]:
                output[y:y+thumb_h, x:x+thumb_w] = thumb
        
        return output
    
    def _generate_summary(self) -> dict:
        """Generate session summary."""
        summary = {
            'poses_completed': self.current_pose_idx,
            'total_poses': len(self.poses),
            'pose_scores': {}
        }
        
        for pose_name, scores in self.session_scores.items():
            summary['pose_scores'][pose_name] = {
                'average': np.mean(scores),
                'best': np.max(scores),
                'worst': np.min(scores)
            }
        
        if self.session_scores:
            all_scores = [s for scores in self.session_scores.values() for s in scores]
            summary['overall_average'] = np.mean(all_scores)
        
        return summary
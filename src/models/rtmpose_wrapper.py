"""RTMPose model wrapper using rtmlib-style implementation."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import urllib.request
import zipfile
import os

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class RTMPoseWrapper:
    """
    Lightweight RTMPose wrapper without mmcv/mmpose dependencies.
    Uses ONNX Runtime for inference.
    """
    
    # Model URLs from OpenMMLab
    MODEL_URLS = {
        't': {
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
            'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
        },
        's': {
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',
            'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-40f1f3a9.zip',
        },
        'm': {
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
            'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
        },
        'l': {
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip',
            'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a6a.zip',
        },
    }
    
    INPUT_SIZES = {
        't': (192, 256),
        's': (192, 256),
        'm': (192, 256),
        'l': (288, 384),
    }
    
    def __init__(
        self,
        size: str = 's',
        device: str = 'cpu',
        weights_dir: str = 'weights',
        det_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None
    ):
        """
        Initialize RTMPose wrapper.
        
        Args:
            size: Model size ('t', 's', 'm', 'l')
            device: 'cpu' or 'cuda'
            weights_dir: Directory to store weights
            det_model_path: Custom detector ONNX path
            pose_model_path: Custom pose model ONNX path
        """
        if ort is None:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        
        self.size = size
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.input_size = self.INPUT_SIZES.get(size, (192, 256))
        
        # Load models
        self.det_session = self._load_model(det_model_path or self._get_model_path('det'))
        self.pose_session = self._load_model(pose_model_path or self._get_model_path('pose'))
        
        # Get input/output names
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.pose_input_name = self.pose_session.get_inputs()[0].name
    
    def _get_model_path(self, model_type: str) -> str:
        """Get or download model path."""
        url = self.MODEL_URLS[self.size][model_type]
        filename = url.split('/')[-1]
        zip_path = self.weights_dir / filename
        onnx_path = self.weights_dir / filename.replace('.zip', '.onnx')
        
        if not onnx_path.exists():
            if not zip_path.exists():
                self._download_model(url, zip_path)
            self._extract_onnx(zip_path, onnx_path)
        
        return str(onnx_path)
    
    def _download_model(self, url: str, save_path: Path):
        """Download model from URL."""
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved to {save_path}")
    
    def _extract_onnx(self, zip_path: Path, onnx_path: Path):
        """Extract ONNX file from zip."""
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.onnx'):
                    with zf.open(name) as src:
                        with open(onnx_path, 'wb') as dst:
                            dst.write(src.read())
                    break
    
    def _load_model(self, model_path: str) -> ort.InferenceSession:
        """Load ONNX model."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        return ort.InferenceSession(model_path, providers=providers)
    
    def detect_persons(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[np.ndarray]:
        """
        Detect persons in image using YOLOX.
        
        Args:
            image: BGR image
            conf_threshold: Confidence threshold
        
        Returns:
            List of bounding boxes [x1, y1, x2, y2, score]
        """
        h, w = image.shape[:2]
        
        # Preprocess for YOLOX
        input_size = (640, 640)
        scale = min(input_size[0] / h, input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalize
        blob = padded.astype(np.float32)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Inference
        outputs = self.det_session.run(None, {self.det_input_name: blob})
        
        # Post-process
        boxes = self._postprocess_detection(outputs[0], scale, (h, w), conf_threshold)
        
        return boxes
    
    def _postprocess_detection(
        self,
        outputs: np.ndarray,
        scale: float,
        orig_size: Tuple[int, int],
        conf_threshold: float
    ) -> List[np.ndarray]:
        """Post-process YOLOX detection outputs."""
        # Simplified post-processing
        # In production, use proper NMS and class filtering
        
        boxes = []
        
        if outputs is None or len(outputs) == 0:
            return boxes
        
        # Assuming outputs shape: [1, N, 85] (x, y, w, h, obj_conf, class_scores...)
        # Filter by confidence and class (person = 0)
        
        predictions = outputs[0]
        
        for pred in predictions:
            obj_conf = pred[4]
            if obj_conf < conf_threshold:
                continue
            
            # Get class scores
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Only keep person class (0)
            if class_id != 0:
                continue
            
            score = obj_conf * class_conf
            if score < conf_threshold:
                continue
            
            # Convert to x1, y1, x2, y2
            cx, cy, w, h = pred[:4]
            x1 = (cx - w / 2) / scale
            y1 = (cy - h / 2) / scale
            x2 = (cx + w / 2) / scale
            y2 = (cy + h / 2) / scale
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_size[1]))
            y1 = max(0, min(y1, orig_size[0]))
            x2 = max(0, min(x2, orig_size[1]))
            y2 = max(0, min(y2, orig_size[0]))
            
            boxes.append(np.array([x1, y1, x2, y2, score]))
        
        return boxes
    
    def estimate_pose(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate pose for single person.
        
        Args:
            image: BGR image
            bbox: Bounding box [x1, y1, x2, y2] or None for full image
        
        Returns:
            keypoints: (17, 2) array of [x, y] coordinates
            scores: (17,) array of confidence scores
        """
        h, w = image.shape[:2]
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]
        else:
            crop = image
            crop_h, crop_w = h, w
            x1, y1 = 0, 0
        
        # Resize to input size
        input_w, input_h = self.input_size
        resized = cv2.resize(crop, (input_w, input_h))
        
        # Normalize
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        blob = (resized.astype(np.float32) - mean) / std
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Inference
        outputs = self.pose_session.run(None, {self.pose_input_name: blob})
        
        # Post-process SimCC outputs
        simcc_x, simcc_y = outputs[0], outputs[1]
        keypoints, scores = self._decode_simcc(simcc_x, simcc_y, input_w, input_h)
        
        # Scale back to original image coordinates
        scale_x = crop_w / input_w
        scale_y = crop_h / input_h
        keypoints[:, 0] = keypoints[:, 0] * scale_x + x1
        keypoints[:, 1] = keypoints[:, 1] * scale_y + y1
        
        return keypoints, scores
    
    def _decode_simcc(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        input_w: int,
        input_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode SimCC coordinate classification outputs."""
        # simcc_x shape: [1, 17, W*2]
        # simcc_y shape: [1, 17, H*2]
        
        simcc_x = simcc_x[0]
        simcc_y = simcc_y[0]
        
        num_keypoints = simcc_x.shape[0]
        keypoints = np.zeros((num_keypoints, 2))
        scores = np.zeros(num_keypoints)
        
        for i in range(num_keypoints):
            x_idx = np.argmax(simcc_x[i])
            y_idx = np.argmax(simcc_y[i])
            
            x_score = np.max(simcc_x[i])
            y_score = np.max(simcc_y[i])
            
            # Convert from SimCC space to image space
            x = x_idx / 2.0
            y = y_idx / 2.0
            
            keypoints[i] = [x, y]
            scores[i] = (x_score + y_score) / 2
        
        return keypoints, scores
    
    def __call__(
        self,
        image: np.ndarray,
        detect_first: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run full pipeline: detection + pose estimation.
        
        Args:
            image: BGR image
            detect_first: Whether to run detection first
        
        Returns:
            List of (keypoints, scores) tuples for each person
        """
        results = []
        
        if detect_first:
            boxes = self.detect_persons(image)
            if not boxes:
                # Full image if no detection
                keypoints, scores = self.estimate_pose(image)
                results.append((keypoints, scores))
            else:
                for box in boxes:
                    keypoints, scores = self.estimate_pose(image, box)
                    results.append((keypoints, scores))
        else:
            keypoints, scores = self.estimate_pose(image)
            results.append((keypoints, scores))
        
        return results
    
    def export_onnx(self, output_path: str):
        """Export pose model to ONNX (already ONNX, just copy)."""
        import shutil
        pose_path = self._get_model_path('pose')
        shutil.copy(pose_path, output_path)
        print(f"Exported to {output_path}")


class SimplePoseEstimator:
    """
    Simplified pose estimator for single person without detection.
    Faster for controlled scenarios (yoga, physical therapy).
    """
    
    def __init__(
        self,
        size: str = 's',
        device: str = 'cpu',
        weights_dir: str = 'weights'
    ):
        """Initialize simple pose estimator."""
        self.wrapper = RTMPoseWrapper(
            size=size,
            device=device,
            weights_dir=weights_dir
        )
        self.input_size = self.wrapper.input_size
    
    def estimate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate pose for single person (assumes person is in frame).
        
        Args:
            image: BGR image
        
        Returns:
            keypoints: (17, 2) array
            scores: (17,) array
        """
        return self.wrapper.estimate_pose(image, bbox=None)
    
    def estimate_with_confidence(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Estimate pose and return keypoints with confidence.
        
        Args:
            image: BGR image
            confidence_threshold: Minimum confidence
        
        Returns:
            keypoints: (17, 3) array with [x, y, confidence]
        """
        keypoints, scores = self.estimate(image)
        
        # Combine into single array
        result = np.zeros((17, 3))
        result[:, :2] = keypoints
        result[:, 2] = scores
        
        # Zero out low confidence keypoints
        result[scores < confidence_threshold, :2] = 0
        
        return result
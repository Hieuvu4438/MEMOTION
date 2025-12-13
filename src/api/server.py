"""FastAPI server for pose assessment API."""

import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Optional
import tempfile
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AssessmentResponse,
    VideoAssessmentResponse,
    RealtimeFeedbackResponse,
    HealthCheckResponse,
    FeedbackSchema,
    JointErrorSchema,
    Keypoint,
)
from ..core.pose_assessor import PoseAssessor
from ..core.video_comparator import VideoComparator


app = FastAPI(
    title="Pose Assessment API",
    description="API for yoga and physical therapy pose assessment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assessor: Optional[PoseAssessor] = None
video_comparator: Optional[VideoComparator] = None
reference_store = {}


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global assessor, video_comparator
    assessor = PoseAssessor(model_size='s', device='cpu', language='vi')
    video_comparator = VideoComparator(model_size='s', device='cpu')


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=assessor is not None,
        version="1.0.0"
    )


@app.post("/api/v1/assess/image", response_model=AssessmentResponse)
async def assess_image(
    user_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    pose_name: Optional[str] = None,
    return_visualization: bool = False
):
    """Assess user's pose against reference image."""
    if assessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        user_bytes = await user_image.read()
        ref_bytes = await reference_image.read()
        
        user_arr = np.frombuffer(user_bytes, np.uint8)
        ref_arr = np.frombuffer(ref_bytes, np.uint8)
        
        user_img = cv2.imdecode(user_arr, cv2.IMREAD_COLOR)
        ref_img = cv2.imdecode(ref_arr, cv2.IMREAD_COLOR)
        
        if user_img is None or ref_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        result = assessor.assess(user_img, ref_img, pose_name=pose_name, visualize=return_visualization)
        
        feedback = FeedbackSchema(
            overall_score=result.feedback.overall_score,
            rating=result.feedback.rating,
            errors=[
                JointErrorSchema(
                    joint_name=e.joint_name,
                    error_type=e.error_type,
                    current_angle=e.current_angle,
                    expected_angle=e.expected_angle,
                    difference=e.difference,
                    severity=e.severity,
                    suggestion=e.suggestion
                ) for e in result.feedback.errors
            ],
            suggestions=result.feedback.suggestions,
            positive_feedback=result.feedback.positive_feedback,
            pose_name=pose_name
        )
        
        vis_base64 = None
        if return_visualization and result.visualized_image is not None:
            _, buffer = cv2.imencode('.jpg', result.visualized_image)
            vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return AssessmentResponse(
            success=True,
            overall_score=result.overall_score,
            feedback=feedback,
            joint_scores=result.comparison.joint_scores,
            error_joints=result.comparison.error_joints,
            visualization_base64=vis_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/reference/upload")
async def upload_reference(
    reference_id: str,
    image: UploadFile = File(...)
):
    """Upload and cache a reference pose."""
    if assessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        img_bytes = await image.read()
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        keypoints = assessor.estimate_pose(img)
        reference_store[reference_id] = {
            'keypoints': keypoints,
            'image': img
        }
        
        return {"success": True, "reference_id": reference_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/assess/with-reference")
async def assess_with_cached_reference(
    reference_id: str,
    user_image: UploadFile = File(...),
    pose_name: Optional[str] = None
):
    """Assess using a pre-cached reference pose."""
    if assessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if reference_id not in reference_store:
        raise HTTPException(status_code=404, detail="Reference not found")
    
    try:
        user_bytes = await user_image.read()
        user_arr = np.frombuffer(user_bytes, np.uint8)
        user_img = cv2.imdecode(user_arr, cv2.IMREAD_COLOR)
        
        if user_img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        ref_kpts = reference_store[reference_id]['keypoints']
        result = assessor.assess_with_reference(user_img, ref_kpts, pose_name=pose_name)
        
        return {
            "success": True,
            "overall_score": result.overall_score,
            "rating": result.feedback.rating,
            "suggestions": result.feedback.suggestions,
            "error_joints": result.comparison.error_joints
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/assess/video", response_model=VideoAssessmentResponse)
async def assess_video(
    user_video: UploadFile = File(...),
    reference_video: UploadFile = File(...),
    use_dtw: bool = True
):
    """Compare user video with reference video."""
    if video_comparator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as user_tmp:
            user_tmp.write(await user_video.read())
            user_path = user_tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as ref_tmp:
            ref_tmp.write(await reference_video.read())
            ref_path = ref_tmp.name
        
        result = video_comparator.compare(user_path, ref_path, use_dtw=use_dtw)
        
        import os
        os.unlink(user_path)
        os.unlink(ref_path)
        
        return VideoAssessmentResponse(
            success=True,
            overall_score=result.overall_score,
            frame_scores=result.frame_scores,
            joint_scores=result.joint_scores,
            error_joints=result.error_joints,
            best_frame=result.best_frame_idx,
            worst_frame=result.worst_frame_idx
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/v1/assess/stream")
async def realtime_assessment(websocket: WebSocket):
    """WebSocket endpoint for real-time pose assessment."""
    await websocket.accept()
    
    if assessor is None:
        await websocket.close(code=1011, reason="Model not loaded")
        return
    
    reference_kpts = None
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'set_reference':
                img_base64 = data.get('image')
                if img_base64:
                    img_bytes = base64.b64decode(img_base64)
                    img_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        reference_kpts = assessor.estimate_pose(img)
                        await websocket.send_json({"type": "reference_set", "success": True})
                continue
            
            if data.get('type') == 'assess':
                img_base64 = data.get('image')
                if img_base64 and reference_kpts is not None:
                    img_bytes = base64.b64decode(img_base64)
                    img_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        score, messages, kpts = assessor.get_realtime_feedback(img, reference_kpts)
                        
                        await websocket.send_json({
                            "type": "feedback",
                            "score": score,
                            "messages": messages,
                            "keypoints": [
                                {"x": float(kpts[i, 0]), "y": float(kpts[i, 1]), "confidence": float(kpts[i, 2])}
                                for i in range(len(kpts))
                            ]
                        })
                        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))


@app.get("/api/v1/poses")
async def list_poses():
    """List available yoga poses."""
    from ..config.pose_config import YOGA_POSES
    
    return {
        "poses": [
            {
                "id": pose_id,
                "name": pose.name,
                "name_vi": pose.name_vi,
                "description": pose.description,
                "difficulty": pose.difficulty,
                "benefits": pose.benefits
            }
            for pose_id, pose in YOGA_POSES.items()
        ]
    }


@app.get("/api/v1/exercises")
async def list_exercises():
    """List available physical therapy exercises."""
    from ..config.pose_config import PHYSICAL_THERAPY_EXERCISES
    
    return {
        "exercises": [
            {
                "id": ex_id,
                "name": ex.name,
                "name_vi": ex.name_vi,
                "description": ex.description,
                "target_joints": ex.target_joints,
                "repetitions": ex.repetitions,
                "hold_seconds": ex.hold_seconds
            }
            for ex_id, ex in PHYSICAL_THERAPY_EXERCISES.items()
        ]
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
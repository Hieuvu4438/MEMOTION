#!/usr/bin/env python3
"""
Example usage of the Pose Assessment System.

This script demonstrates different ways to use the system:
1. Single image assessment
2. Real-time camera assessment
3. Video comparison
4. Guided exercise session
"""

import cv2
import numpy as np
from pathlib import Path


def example_1_single_image():
    """Example 1: Assess single image against reference."""
    print("\n" + "="*50)
    print("Example 1: Single Image Assessment")
    print("="*50)
    
    from src.core.pose_assessor import PoseAssessor
    
    # Initialize assessor
    assessor = PoseAssessor(
        model_size='s',      # Use small model for mobile
        device='cpu',        # Use 'cuda' for GPU
        language='vi'        # Vietnamese feedback
    )
    
    # For demo, create synthetic images
    # In real usage, load actual images
    print("\nCreating synthetic test data...")
    user_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    reference_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Note: Using random images for demo. Use real images for actual assessment.")
    
    # Assess
    # result = assessor.assess(
    #     user_image='path/to/user_image.jpg',
    #     reference_image='path/to/reference.jpg',
    #     pose_name='warrior_i',
    #     visualize=True
    # )
    
    # print(f"\nOverall Score: {result.overall_score:.1f}%")
    # print(f"Rating: {result.feedback.rating}")
    # print(f"\nSuggestions:")
    # for suggestion in result.feedback.suggestions:
    #     print(f"  - {suggestion}")
    
    print("\nTo use with real images:")
    print("""
    result = assessor.assess(
        user_image='path/to/user_image.jpg',
        reference_image='path/to/reference.jpg',
        pose_name='warrior_i',
        visualize=True
    )
    
    print(f"Score: {result.overall_score:.1f}%")
    for suggestion in result.feedback.suggestions:
        print(f"  - {suggestion}")
    """)


def example_2_realtime_camera():
    """Example 2: Real-time assessment from camera."""
    print("\n" + "="*50)
    print("Example 2: Real-time Camera Assessment")
    print("="*50)
    
    print("""
    from src.core.realtime_assessor import RealtimeAssessor, RealtimeConfig
    
    # Configure
    config = RealtimeConfig(
        show_skeleton=True,
        show_reference=True,
        show_feedback=True,
        mirror=True,
        target_fps=30.0
    )
    
    # Initialize with reference pose
    assessor = RealtimeAssessor(
        reference_source='path/to/reference_pose.jpg',
        model_size='s',
        config=config
    )
    
    # Start camera assessment
    assessor.start(camera_id=0)
    
    # Controls:
    # - Press 'q' or ESC to quit
    # - Press 'r' to capture new reference from current frame
    # - Press 's' to save screenshot
    """)


def example_3_video_comparison():
    """Example 3: Compare video with reference video."""
    print("\n" + "="*50)
    print("Example 3: Video-to-Video Comparison")
    print("="*50)
    
    print("""
    from src.core.video_comparator import VideoComparator
    
    # Initialize
    comparator = VideoComparator(model_size='s')
    
    # Compare videos
    result = comparator.compare(
        user_video='path/to/user_video.mp4',
        reference_video='path/to/reference_video.mp4',
        use_dtw=True,  # Dynamic Time Warping for temporal alignment
        output_video='comparison_output.mp4'  # Optional: save comparison
    )
    
    print(f"Overall Score: {result.overall_score:.1f}%")
    print(f"Best frame: #{result.best_frame_idx} ({max(result.frame_scores):.1f}%)")
    print(f"Worst frame: #{result.worst_frame_idx} ({min(result.frame_scores):.1f}%)")
    
    # Generate report
    comparator.generate_report(result, 'assessment_report.html')
    """)


def example_4_guided_session():
    """Example 4: Guided exercise session."""
    print("\n" + "="*50)
    print("Example 4: Guided Exercise Session")
    print("="*50)
    
    print("""
    from src.core.pose_assessor import PoseAssessor
    from src.core.realtime_assessor import GuidedExerciseSession
    
    # Define exercise sequence
    poses = [
        ('Warrior I', 'data/reference_poses/warrior_i.jpg'),
        ('Warrior II', 'data/reference_poses/warrior_ii.jpg'),
        ('Tree Pose', 'data/reference_poses/tree_pose.jpg'),
        ('Downward Dog', 'data/reference_poses/downward_dog.jpg'),
    ]
    
    # Initialize
    assessor = PoseAssessor(model_size='s')
    session = GuidedExerciseSession(
        assessor=assessor,
        poses=poses,
        hold_time=10.0,      # Hold each pose for 10 seconds
        transition_time=5.0  # 5 seconds between poses
    )
    
    # Run session
    summary = session.run(camera_id=0)
    
    # View results
    print(f"Completed: {summary['poses_completed']}/{summary['total_poses']}")
    print(f"Overall average: {summary.get('overall_average', 0):.1f}%")
    
    for pose_name, scores in summary['pose_scores'].items():
        print(f"  {pose_name}: avg={scores['average']:.1f}%, best={scores['best']:.1f}%")
    """)


def example_5_api_usage():
    """Example 5: Using the REST API."""
    print("\n" + "="*50)
    print("Example 5: REST API Usage")
    print("="*50)
    
    print("""
    # Start the server:
    # python -m src.api.server --port 8000
    
    # Or programmatically:
    from src.api.server import run_server
    run_server(port=8000)
    
    # Then use the API:
    
    import requests
    
    # Health check
    response = requests.get('http://localhost:8000/health')
    print(response.json())
    
    # Assess image
    files = {
        'user_image': open('user.jpg', 'rb'),
        'reference_image': open('reference.jpg', 'rb')
    }
    response = requests.post(
        'http://localhost:8000/api/v1/assess/image',
        files=files,
        params={'pose_name': 'warrior_i', 'return_visualization': True}
    )
    result = response.json()
    print(f"Score: {result['overall_score']}%")
    
    # WebSocket for real-time (using websockets library):
    import asyncio
    import websockets
    import base64
    
    async def realtime_assess():
        async with websockets.connect('ws://localhost:8000/api/v1/assess/stream') as ws:
            # Set reference
            with open('reference.jpg', 'rb') as f:
                ref_b64 = base64.b64encode(f.read()).decode()
            
            await ws.send(json.dumps({
                'type': 'set_reference',
                'image': ref_b64
            }))
            
            # Send frames and receive feedback
            while True:
                frame = capture_frame()  # Your camera capture
                frame_b64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
                
                await ws.send(json.dumps({
                    'type': 'assess',
                    'image': frame_b64
                }))
                
                feedback = await ws.recv()
                print(json.loads(feedback))
    """)


def example_6_mobile_export():
    """Example 6: Export for mobile deployment."""
    print("\n" + "="*50)
    print("Example 6: Mobile Deployment")
    print("="*50)
    
    print("""
    # The RTMPose models are already in ONNX format.
    # For mobile deployment, you can use:
    
    # 1. TensorFlow Lite (Android/iOS)
    # Convert ONNX to TFLite:
    # pip install onnx-tf
    # python -m onnx_tf.convert -i weights/rtmpose-s.onnx -o weights/rtmpose-s.pb
    # Then convert to TFLite using TensorFlow
    
    # 2. ONNX Runtime Mobile (Android/iOS)
    # Use the ONNX files directly with ONNX Runtime Mobile SDK
    
    # 3. ncnn (Android/iOS - recommended for speed)
    # Convert ONNX to ncnn using ncnn tools
    
    # 4. CoreML (iOS only)
    # pip install coremltools
    # import coremltools as ct
    # model = ct.converters.onnx.convert(model='weights/rtmpose-s.onnx')
    # model.save('weights/rtmpose-s.mlmodel')
    
    # The comparison and feedback logic can be ported to:
    # - Kotlin/Java for Android
    # - Swift for iOS
    # - Or use a thin backend API
    """)


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("#  POSE ASSESSMENT SYSTEM - USAGE EXAMPLES")
    print("#"*60)
    
    example_1_single_image()
    example_2_realtime_camera()
    example_3_video_comparison()
    example_4_guided_session()
    example_5_api_usage()
    example_6_mobile_export()
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("="*50)
    print("\nTo get started:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download weights: python scripts/download_weights.py")
    print("3. Run examples with actual images/videos")


if __name__ == '__main__':
    main()
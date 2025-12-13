# 🧘 Yoga Therapy - Pose Assessment System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)

**Hệ thống đánh giá tư thế yoga và vật lý trị liệu sử dụng AI**

[English](#english) | [Tiếng Việt](#tiếng-việt)

</div>

---

## Tiếng Việt

### 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [API Reference](#api-reference)
- [Cơ chế hoạt động](#cơ-chế-hoạt-động)

---

### 🎯 Giới thiệu

**Yoga Therapy** là hệ thống AI giúp người dùng tập yoga và vật lý trị liệu đúng cách bằng cách:

1. **Phát hiện tư thế** - Sử dụng RTMPose để phát hiện 17 điểm keypoints trên cơ thể
2. **So sánh với mẫu** - So sánh tư thế người dùng với tư thế chuẩn
3. **Phản hồi real-time** - Đưa ra gợi ý điều chỉnh bằng tiếng Việt/Anh
4. **Báo cáo chi tiết** - Phân tích lỗi và theo dõi tiến độ

---

### ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 🎥 **Real-time Assessment** | Đánh giá tư thế trực tiếp từ webcam |
| 📹 **Video Guided** | So sánh đồng bộ với video bài tập mẫu |
| 🖼️ **Image Comparison** | So sánh 2 ảnh tĩnh |
| 📊 **Video Analysis** | Phân tích và so sánh 2 video |
| 🌐 **REST API** | API server cho tích hợp ứng dụng |
| 🇻🇳 **Multi-language** | Hỗ trợ tiếng Việt và tiếng Anh |
| 📱 **Mobile Ready** | Chuẩn bị cho Android/iOS |

---

### 🚀 Cài đặt

#### Yêu cầu hệ thống

- Python 3.8+
- Webcam (cho real-time)
- RAM: 4GB+ (khuyến nghị 8GB)
- GPU: Optional (tăng tốc xử lý)

#### Cài đặt cơ bản

```bash
# Clone repository
git clone https://github.com/yourusername/yoga_therapy.git
cd yoga_therapy

# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate

# Cài đặt package
pip install -e .
```

#### Cài đặt với GPU (CUDA)

```bash
pip install -e ".[gpu]"
```

#### Cài đặt đầy đủ (bao gồm DTW)

```bash
pip install -e ".[dtw]"
```

#### Download model weights

```bash
python scripts/download_weights.py
```

---

### 📁 Cấu trúc dự án

```
yoga_therapy/
│
├── 📄 README.md              # Tài liệu này
├── 📄 setup.py               # Cài đặt package
├── 📄 readme.py              # Tài liệu dạng Python
├── 📄 usage_examples.py      # Ví dụ sử dụng
│
├── 📂 src/                   # Source code chính
│   ├── 📂 api/               # REST API
│   │   ├── server.py         # FastAPI server
│   │   └── schemas.py        # Pydantic schemas
│   │
│   ├── 📂 config/            # Cấu hình
│   │   ├── model_config.py   # Cấu hình model AI
│   │   └── pose_config.py    # Cấu hình pose (joints, weights)
│   │
│   ├── 📂 core/              # Logic xử lý chính
│   │   ├── pose_comparator.py      # So sánh 2 poses
│   │   ├── pose_assessor.py        # Đánh giá tổng thể
│   │   ├── feedback_generator.py   # Tạo phản hồi
│   │   ├── realtime_assessor.py    # Đánh giá real-time
│   │   ├── video_comparator.py     # So sánh video
│   │   └── video_guided_assessor.py # Video hướng dẫn
│   │
│   ├── 📂 models/            # AI Models
│   │   └── rtmpose_wrapper.py # Wrapper cho RTMPose
│   │
│   └── 📂 utils/             # Tiện ích
│       ├── math_utils.py     # Hàm toán học
│       ├── video_utils.py    # Xử lý video
│       └── visualization.py  # Vẽ skeleton
│
├── 📂 data/                  # Dữ liệu
│   ├── 📂 reference_poses/   # Poses chuẩn (.npy, .jpg)
│   └── 📂 test_videos/       # Video test
│
├── 📂 weights/               # Model weights (.onnx)
├── 📂 tests/                 # Unit tests
├── 📂 scripts/               # Scripts tiện ích
├── 📂 examples/              # Code ví dụ
├── 📂 notebooks/             # Jupyter notebooks
└── 📂 mobile/                # Mobile apps (Android/iOS)
```

---

### 📖 Hướng dẫn sử dụng

#### 1️⃣ So sánh 2 ảnh đơn giản

```python
import numpy as np
from src.core.pose_comparator import PoseComparator
from src.core.feedback_generator import FeedbackGenerator

# Tạo sample keypoints (17 điểm COCO format)
# Format: [x, y, confidence] cho mỗi keypoint
user_keypoints = np.array([
    [100, 50, 0.9],   # 0: nose
    [95, 45, 0.8],    # 1: left_eye
    [105, 45, 0.8],   # 2: right_eye
    [90, 50, 0.7],    # 3: left_ear
    [110, 50, 0.7],   # 4: right_ear
    [80, 100, 0.9],   # 5: left_shoulder
    [120, 100, 0.9],  # 6: right_shoulder
    [70, 150, 0.9],   # 7: left_elbow
    [130, 150, 0.9],  # 8: right_elbow
    [65, 200, 0.8],   # 9: left_wrist
    [135, 200, 0.8],  # 10: right_wrist
    [85, 200, 0.9],   # 11: left_hip
    [115, 200, 0.9],  # 12: right_hip
    [80, 280, 0.9],   # 13: left_knee
    [120, 280, 0.9],  # 14: right_knee
    [75, 350, 0.8],   # 15: left_ankle
    [125, 350, 0.8],  # 16: right_ankle
], dtype=np.float32)

# Reference pose (tư thế chuẩn)
reference_keypoints = user_keypoints.copy()

# So sánh
comparator = PoseComparator(method='weighted')
result = comparator.compare(user_keypoints, reference_keypoints)

print(f"Điểm tổng: {result.overall_score:.1f}%")
print(f"Phương pháp: {result.method}")
print(f"Khớp lỗi: {result.error_joints}")

# Tạo feedback tiếng Việt
feedback_gen = FeedbackGenerator(language='vi')
feedback = feedback_gen.generate_feedback(
    result, user_keypoints, reference_keypoints
)

print(f"\nĐánh giá: {feedback.rating}")
print(f"Nhận xét: {feedback.overall_message}")
for error in feedback.errors:
    print(f"  - {error.joint_name}: {error.suggestion}")
```

#### 2️⃣ Real-time từ Webcam

```python
from src.core.realtime_assessor import RealtimeAssessor

# Khởi tạo với ảnh reference
assessor = RealtimeAssessor(
    model_size='s',
    language='vi'
)

# Load pose chuẩn
assessor.load_reference('data/reference_poses/warrior_pose.jpg')

# Bắt đầu đánh giá real-time
# Nhấn 'q' để thoát
assessor.start(camera_id=0)

# Xem kết quả
summary = assessor.get_session_summary()
print(f"Điểm trung bình: {summary['average_score']:.1f}%")
```

#### 3️⃣ So sánh đồng bộ với Video mẫu

```python
from src.core.video_guided_assessor import start_video_guided_session

# Chạy session với video yoga 30s
start_video_guided_session(
    reference_video='data/test_videos/warrior_pose_30s.mp4',
    camera_id=0,
    loop=True,           # Lặp lại video khi hết
    language='vi',
    save_video='my_session.mp4'  # Lưu video session
)

# Controls:
# Q/ESC - Thoát
# R     - Restart
# SPACE - Pause/Resume
```

#### 4️⃣ So sánh 2 Video (Offline)

```python
from src.core.video_comparator import VideoComparator

comparator = VideoComparator(model_size='s')

result = comparator.compare(
    user_video='user_exercise.mp4',
    reference_video='data/test_videos/reference.mp4',
    use_dtw=True  # Dynamic Time Warping để đồng bộ tốc độ
)

print(f"Điểm trung bình: {result['overall_score']:.1f}%")
print(f"Frame tốt nhất: {result['best_frame']}")
print(f"Frame cần cải thiện: {result['worst_frame']}")
```

#### 5️⃣ Sử dụng API Server

```bash
# Khởi động server
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

```python
import requests
import base64

# Đọc ảnh và encode base64
with open('user_pose.jpg', 'rb') as f:
    user_image = base64.b64encode(f.read()).decode()

with open('reference_pose.jpg', 'rb') as f:
    ref_image = base64.b64encode(f.read()).decode()

# Gọi API
response = requests.post(
    'http://localhost:8000/compare',
    json={
        'user_image': user_image,
        'reference_image': ref_image,
        'language': 'vi'
    }
)

result = response.json()
print(f"Score: {result['score']}%")
print(f"Feedback: {result['feedback']}")
```

#### 6️⃣ Guided Exercise Session (Nhiều tư thế)

```python
from src.core.realtime_assessor import GuidedExerciseSession

# Định nghĩa các tư thế trong bài tập
poses = [
    {'name': 'mountain_pose', 'duration': 10, 'reference': 'poses/mountain.npy'},
    {'name': 'warrior_1', 'duration': 15, 'reference': 'poses/warrior1.npy'},
    {'name': 'warrior_2', 'duration': 15, 'reference': 'poses/warrior2.npy'},
    {'name': 'tree_pose', 'duration': 20, 'reference': 'poses/tree.npy'},
]

session = GuidedExerciseSession(
    poses=poses,
    language='vi',
    model_size='s'
)

# Bắt đầu session
session.start(camera_id=0)

# Xem báo cáo
report = session.get_report()
print(f"Tổng điểm: {report['total_score']:.1f}%")
for pose_result in report['pose_results']:
    print(f"  {pose_result['name']}: {pose_result['score']:.1f}%")
```

---

### 🔌 API Reference

#### Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/compare` | So sánh 2 poses từ ảnh |
| `POST` | `/assess` | Đánh giá 1 pose |
| `GET` | `/poses` | Lấy danh sách poses chuẩn |
| `GET` | `/health` | Health check |
| `WS` | `/ws/realtime` | WebSocket cho real-time |

#### Request/Response Examples

**POST /compare**

```json
// Request
{
    "user_image": "base64_encoded_image",
    "reference_image": "base64_encoded_image",
    "method": "weighted",
    "language": "vi"
}

// Response
{
    "score": 85.5,
    "rating": "good",
    "error_joints": ["left_elbow", "right_knee"],
    "feedback": {
        "overall": "Tốt lắm! Cần điều chỉnh nhẹ.",
        "suggestions": [
            "Điều chỉnh góc khuỷu tay trái",
            "Gập gối phải hơn"
        ]
    },
    "joint_scores": {
        "left_elbow": 65.0,
        "right_elbow": 90.0,
        "left_knee": 88.0,
        "right_knee": 58.0
    }
}
```

---

### ⚙️ Cơ chế hoạt động

#### Flowchart

```
┌─────────────────┐     ┌─────────────────┐
│  VIDEO/CAMERA   │     │  VIDEO MẪU      │
│  (User Input)   │     │  (Reference)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           RTMPose Model                  │
│    Detect 17 Keypoints (COCO format)    │
└────────┬───────────────────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ User Keypoints  │     │ Ref Keypoints   │
│   (17, 3)       │     │   (17, 3)       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   POSE COMPARATOR     │
         │                       │
         │  ┌─────────────────┐  │
         │  │ Cosine (20%)    │  │
         │  │ Angle (50%)     │  │
         │  │ OKS (30%)       │  │
         │  └─────────────────┘  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  FEEDBACK GENERATOR   │
         │  - Phân tích lỗi      │
         │  - Tạo gợi ý (VI/EN)  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    OUTPUT             │
         │  - Score: 85%         │
         │  - Errors: [...]      │
         │  - Suggestions: [...] │
         └───────────────────────┘
```

#### 17 Keypoints (COCO Format)

```
         0 (nose)
        /|\
       1 2 (eyes)
      / | \
     3  |  4 (ears)
        |
    5───┼───6 (shoulders)
    |   |   |
    7   |   8 (elbows)
    |   |   |
    9   |  10 (wrists)
        |
   11───┼──12 (hips)
    |   |   |
   13   |  14 (knees)
    |   |   |
   15   |  16 (ankles)
```

#### Phương pháp so sánh

| Phương pháp | Weight | Mô tả |
|-------------|--------|-------|
| **Cosine** | 20% | So sánh vector toàn bộ pose (nhanh, scale-invariant) |
| **Angle** | 50% | So sánh góc từng khớp (trực quan, dễ hiểu) |
| **OKS** | 30% | Object Keypoint Similarity - chuẩn COCO |

#### Phân loại lỗi

| Error Type | Điều kiện | Mô tả |
|------------|-----------|-------|
| `too_bent` | user_angle < ref_angle - 5° | Gập quá nhiều |
| `too_straight` | user_angle > ref_angle + 5° | Duỗi quá thẳng |
| `misaligned` | Còn lại | Lệch vị trí |

#### Mức độ nghiêm trọng

| Severity | Chênh lệch góc | Màu hiển thị |
|----------|----------------|--------------|
| `minor` | 10° - 19° | 🟡 Vàng |
| `moderate` | 20° - 29° | 🟠 Cam |
| `severe` | ≥ 30° | 🔴 Đỏ |

#### Bảng đánh giá

| Score | Rating | Message (VI) |
|-------|--------|--------------|
| 90-100% | ⭐⭐⭐⭐⭐ excellent | Xuất sắc! Tư thế rất chuẩn |
| 75-89% | ⭐⭐⭐⭐ good | Tốt lắm! Cần điều chỉnh nhẹ |
| 60-74% | ⭐⭐⭐ needs_improvement | Cần cải thiện một số điểm |
| 0-59% | ⭐⭐ poor | Cần điều chỉnh nhiều |

---

### 🧪 Testing

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy với coverage
pytest tests/ -v --cov=src --cov-report=html

# Chạy test cụ thể
pytest tests/test_pose_comparator.py -v
```

---

### 🛠️ Troubleshooting

#### Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `Model not found` | Chưa download weights | `python scripts/download_weights.py` |
| `Camera not found` | Webcam không kết nối | Kiểm tra camera ID |
| `CUDA out of memory` | GPU không đủ RAM | Dùng `device='cpu'` hoặc model nhỏ hơn |
| `Import error` | Thiếu dependencies | `pip install -e .` |

#### Performance Tips

1. **Dùng model size nhỏ** (`'t'` hoặc `'s'`) cho real-time
2. **Giảm resolution** camera xuống 640x480
3. **Bật GPU** nếu có: `pip install -e ".[gpu]"`
4. **Skip frames** nếu FPS thấp

---

### 📜 License

MIT License - Xem file [LICENSE](LICENSE) để biết chi tiết.

---

### 🤝 Contributing

1. Fork repository
2. Tạo branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

### 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Project Link**: https://github.com/yourusername/yoga_therapy

---

## English

### 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference-1)
- [How It Works](#how-it-works)

### 🎯 Introduction

**Yoga Therapy** is an AI-powered system that helps users practice yoga and physical therapy correctly by:

1. **Pose Detection** - Using RTMPose to detect 17 body keypoints
2. **Comparison** - Comparing user's pose with reference poses
3. **Real-time Feedback** - Providing correction suggestions in English/Vietnamese
4. **Detailed Reports** - Analyzing errors and tracking progress

### ✨ Features

- 🎥 **Real-time Assessment** from webcam
- 📹 **Video Guided** sync with reference video
- 🖼️ **Image Comparison** between two static images
- 📊 **Video Analysis** compare two videos
- 🌐 **REST API** for app integration
- 🇺🇸🇻🇳 **Multi-language** support

### 🚀 Installation

```bash
# Clone and install
git clone https://github.com/yourusername/yoga_therapy.git
cd yoga_therapy
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e .

# Download model
python scripts/download_weights.py
```

### 📖 Usage

#### Quick Start

```python
from src.core.video_guided_assessor import start_video_guided_session

# Start video-guided session
start_video_guided_session(
    reference_video='data/test_videos/yoga_30s.mp4',
    camera_id=0,
    language='en',
    loop=True
)
```

#### Real-time from Webcam

```python
from src.core.realtime_assessor import RealtimeAssessor

assessor = RealtimeAssessor(model_size='s', language='en')
assessor.load_reference('data/reference_poses/warrior.jpg')
assessor.start(camera_id=0)
```

### ⚙️ How It Works

1. **Input**: Video/Camera frame → RTMPose → 17 keypoints
2. **Compare**: User keypoints vs Reference keypoints
3. **Methods**: Cosine (20%) + Angle (50%) + OKS (30%)
4. **Output**: Score + Error joints + Suggestions

### 📜 License

MIT License
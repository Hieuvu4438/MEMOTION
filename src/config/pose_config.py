"""Pose configurations and keypoint definitions."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# COCO Keypoint Format (17 keypoints)
COCO_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

KEYPOINT_INDEX = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),     # nose to eyes
    (1, 3), (2, 4),     # eyes to ears
    (5, 6),             # shoulders
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso
    (11, 12),           # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]

# Joint angles to calculate for pose comparison
# Format: (joint_name, (point_a_idx, point_b_idx, point_c_idx))
# Angle is calculated at point_b
JOINT_ANGLES = {
    'left_elbow': (5, 7, 9),      # shoulder -> elbow -> wrist
    'right_elbow': (6, 8, 10),    # shoulder -> elbow -> wrist
    'left_shoulder': (7, 5, 11),  # elbow -> shoulder -> hip
    'right_shoulder': (8, 6, 12), # elbow -> shoulder -> hip
    'left_hip': (5, 11, 13),      # shoulder -> hip -> knee
    'right_hip': (6, 12, 14),     # shoulder -> hip -> knee
    'left_knee': (11, 13, 15),    # hip -> knee -> ankle
    'right_knee': (12, 14, 16),   # hip -> knee -> ankle
}

# Body parts for weighted comparison
BODY_PARTS = {
    'head': [0, 1, 2, 3, 4],
    'torso': [5, 6, 11, 12],
    'left_arm': [5, 7, 9],
    'right_arm': [6, 8, 10],
    'left_leg': [11, 13, 15],
    'right_leg': [12, 14, 16],
}

# Weight for each keypoint in similarity calculation
# Higher weight = more important for pose assessment
KEYPOINT_WEIGHTS = {
    'nose': 0.5,
    'left_eye': 0.3,
    'right_eye': 0.3,
    'left_ear': 0.3,
    'right_ear': 0.3,
    'left_shoulder': 1.0,
    'right_shoulder': 1.0,
    'left_elbow': 1.0,
    'right_elbow': 1.0,
    'left_wrist': 0.8,
    'right_wrist': 0.8,
    'left_hip': 1.0,
    'right_hip': 1.0,
    'left_knee': 1.0,
    'right_knee': 1.0,
    'left_ankle': 0.8,
    'right_ankle': 0.8,
}

# OKS sigmas for COCO keypoints (from COCO eval)
OKS_SIGMAS = [
    0.026,  # nose
    0.025,  # left_eye
    0.025,  # right_eye
    0.035,  # left_ear
    0.035,  # right_ear
    0.079,  # left_shoulder
    0.079,  # right_shoulder
    0.072,  # left_elbow
    0.072,  # right_elbow
    0.062,  # left_wrist
    0.062,  # right_wrist
    0.107,  # left_hip
    0.107,  # right_hip
    0.087,  # left_knee
    0.087,  # right_knee
    0.089,  # left_ankle
    0.089,  # right_ankle
]


@dataclass
class YogaPoseDefinition:
    """Definition of a yoga pose with expected angles."""
    name: str
    name_vi: str
    description: str
    expected_angles: Dict[str, Tuple[float, float]]  # joint -> (min_angle, max_angle)
    key_points: List[str]  # Critical points to check
    difficulty: str
    benefits: List[str]


YOGA_POSES = {
    'warrior_i': YogaPoseDefinition(
        name='Warrior I',
        name_vi='Tư thế Chiến binh I',
        description='Standing pose with arms raised overhead',
        expected_angles={
            'left_knee': (85, 100),
            'right_knee': (160, 180),
            'left_shoulder': (160, 180),
            'right_shoulder': (160, 180),
        },
        key_points=['left_knee', 'right_knee', 'left_shoulder', 'right_shoulder'],
        difficulty='beginner',
        benefits=['Strengthens legs', 'Opens hips', 'Stretches chest']
    ),
    'warrior_ii': YogaPoseDefinition(
        name='Warrior II',
        name_vi='Tư thế Chiến binh II',
        description='Standing pose with arms extended horizontally',
        expected_angles={
            'left_knee': (85, 100),
            'right_knee': (160, 180),
            'left_shoulder': (85, 100),
            'right_shoulder': (85, 100),
            'left_elbow': (170, 180),
            'right_elbow': (170, 180),
        },
        key_points=['left_knee', 'left_shoulder', 'right_shoulder'],
        difficulty='beginner',
        benefits=['Builds stamina', 'Opens hips', 'Strengthens legs']
    ),
    'tree_pose': YogaPoseDefinition(
        name='Tree Pose',
        name_vi='Tư thế Cây',
        description='Balance on one leg with other foot on inner thigh',
        expected_angles={
            'left_knee': (170, 180),
            'right_knee': (40, 90),
            'left_hip': (160, 180),
        },
        key_points=['left_knee', 'right_knee', 'left_hip'],
        difficulty='beginner',
        benefits=['Improves balance', 'Strengthens legs', 'Opens hips']
    ),
    'downward_dog': YogaPoseDefinition(
        name='Downward Dog',
        name_vi='Tư thế Chó cúi mặt',
        description='Inverted V shape with hands and feet on ground',
        expected_angles={
            'left_shoulder': (160, 180),
            'right_shoulder': (160, 180),
            'left_hip': (70, 100),
            'right_hip': (70, 100),
            'left_knee': (160, 180),
            'right_knee': (160, 180),
        },
        key_points=['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
        difficulty='beginner',
        benefits=['Stretches spine', 'Strengthens arms', 'Calms mind']
    ),
    'cobra_pose': YogaPoseDefinition(
        name='Cobra Pose',
        name_vi='Tư thế Rắn hổ mang',
        description='Lying face down with chest lifted',
        expected_angles={
            'left_elbow': (120, 160),
            'right_elbow': (120, 160),
            'left_hip': (160, 180),
            'right_hip': (160, 180),
        },
        key_points=['left_elbow', 'right_elbow'],
        difficulty='beginner',
        benefits=['Strengthens spine', 'Opens chest', 'Stretches abdomen']
    ),
}


@dataclass
class PhysicalTherapyExercise:
    """Definition of a physical therapy exercise."""
    name: str
    name_vi: str
    description: str
    target_joints: List[str]
    expected_rom: Dict[str, Tuple[float, float]]  # Range of motion
    repetitions: int
    hold_seconds: int
    contraindications: List[str]


PHYSICAL_THERAPY_EXERCISES = {
    'knee_extension': PhysicalTherapyExercise(
        name='Seated Knee Extension',
        name_vi='Duỗi gối khi ngồi',
        description='Straighten knee while seated',
        target_joints=['left_knee', 'right_knee'],
        expected_rom={'knee': (0, 180)},
        repetitions=10,
        hold_seconds=5,
        contraindications=['Acute knee injury', 'Recent knee surgery']
    ),
    'shoulder_flexion': PhysicalTherapyExercise(
        name='Shoulder Flexion',
        name_vi='Gập vai',
        description='Raise arm forward and up',
        target_joints=['left_shoulder', 'right_shoulder'],
        expected_rom={'shoulder': (0, 180)},
        repetitions=10,
        hold_seconds=3,
        contraindications=['Frozen shoulder', 'Acute shoulder injury']
    ),
    'hip_abduction': PhysicalTherapyExercise(
        name='Hip Abduction',
        name_vi='Dạng háng',
        description='Move leg away from body while lying down',
        target_joints=['left_hip', 'right_hip'],
        expected_rom={'hip': (0, 45)},
        repetitions=10,
        hold_seconds=3,
        contraindications=['Hip replacement', 'Acute hip injury']
    ),
}


# Thresholds for error detection
ERROR_THRESHOLDS = {
    'angle_tolerance': 15.0,        # degrees
    'position_tolerance': 0.1,      # normalized distance
    'confidence_threshold': 0.5,    # minimum confidence to consider keypoint
    'similarity_good': 0.85,        # above this = good pose
    'similarity_acceptable': 0.70,  # above this = acceptable
    'similarity_poor': 0.50,        # below this = needs correction
}


# Feedback messages
FEEDBACK_MESSAGES = {
    'left_elbow': {
        'too_bent': 'Duỗi thẳng khuỷu tay trái hơn',
        'too_straight': 'Gập khuỷu tay trái nhẹ hơn',
    },
    'right_elbow': {
        'too_bent': 'Duỗi thẳng khuỷu tay phải hơn',
        'too_straight': 'Gập khuỷu tay phải nhẹ hơn',
    },
    'left_shoulder': {
        'too_low': 'Nâng vai trái cao hơn',
        'too_high': 'Hạ vai trái xuống',
    },
    'right_shoulder': {
        'too_low': 'Nâng vai phải cao hơn',
        'too_high': 'Hạ vai phải xuống',
    },
    'left_knee': {
        'too_bent': 'Duỗi thẳng gối trái hơn',
        'too_straight': 'Gập gối trái nhẹ hơn',
    },
    'right_knee': {
        'too_bent': 'Duỗi thẳng gối phải hơn',
        'too_straight': 'Gập gối phải nhẹ hơn',
    },
    'left_hip': {
        'too_bent': 'Mở rộng hông trái',
        'too_straight': 'Gập hông trái',
    },
    'right_hip': {
        'too_bent': 'Mở rộng hông phải',
        'too_straight': 'Gập hông phải',
    },
}
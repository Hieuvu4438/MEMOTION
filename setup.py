"""Setup script for pose_assessment package."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pose_assessment",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Pose assessment system for yoga and physical therapy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pose_assessment",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "onnxruntime>=1.12.0",
        "fastapi>=0.85.0",
        "uvicorn>=0.18.0",
        "pydantic>=1.10.0",
        "pillow>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.12.0",
        ],
        "dtw": [
            "dtaidistance>=2.3.0",
            "fastdtw>=0.3.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "pose-assess=src.api.server:run_server",
            "download-weights=scripts.download_weights:main",
        ],
    },
)
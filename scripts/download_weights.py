#!/usr/bin/env python3
"""Download model weights."""

import urllib.request
import zipfile
from pathlib import Path
import sys


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


def download_file(url: str, save_path: Path):
    """Download file with progress."""
    print(f"Downloading: {url}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, save_path, progress_hook)
    print(f"\n  Saved to: {save_path}")


def extract_onnx(zip_path: Path, output_dir: Path):
    """Extract ONNX file from zip."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.onnx'):
                onnx_name = Path(name).name
                output_path = output_dir / onnx_name
                
                with zf.open(name) as src:
                    with open(output_path, 'wb') as dst:
                        dst.write(src.read())
                
                print(f"  Extracted: {output_path}")
                return output_path
    return None


def download_models(model_size: str = 's', weights_dir: str = 'weights'):
    """Download models for specified size."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    if model_size not in MODEL_URLS:
        print(f"Unknown model size: {model_size}")
        print(f"Available: {list(MODEL_URLS.keys())}")
        return
    
    urls = MODEL_URLS[model_size]
    
    print(f"\nDownloading RTMPose-{model_size} models...")
    print("=" * 50)
    
    for model_type, url in urls.items():
        filename = url.split('/')[-1]
        zip_path = weights_dir / filename
        
        if not zip_path.exists():
            download_file(url, zip_path)
        else:
            print(f"Already exists: {zip_path}")
        
        onnx_path = weights_dir / filename.replace('.zip', '.onnx')
        if not onnx_path.exists():
            extract_onnx(zip_path, weights_dir)
    
    print("\n✓ Download complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download RTMPose model weights')
    parser.add_argument('--size', '-s', default='s', choices=['t', 's', 'm', 'l'],
                       help='Model size (default: s)')
    parser.add_argument('--output', '-o', default='weights',
                       help='Output directory (default: weights)')
    parser.add_argument('--all', action='store_true',
                       help='Download all model sizes')
    
    args = parser.parse_args()
    
    if args.all:
        for size in MODEL_URLS.keys():
            download_models(size, args.output)
    else:
        download_models(args.size, args.output)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Script 1: Download Pre-trained YOLOv5 Model

This script downloads a pre-trained YOLOv5 model from the Ultralytics repository.
We use YOLOv5s (small) as it provides a good balance between accuracy and speed.
"""

import os
import sys
import torch 
'core python system utilites and pytorch framework for model loading and saving'
from pathlib import Path 
'object oriented path manipulation'
import requests
from tqdm import tqdm
'reqests handles http fallbacks downloads and tqdm provides progress bar for downloads'


def download_yolov5_model(
    model_name: str = 'yolov5s',
    output_dir: str = 'models',
    force_reload: bool = False
) -> str:
    """
    Download pre-trained YOLOv5 model from Ultralytics
    
    Args:
        model_name: YOLOv5 variant (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
        output_dir: Directory to save the model
        force_reload: Force re-download even if model exists
    
    Returns:
        Path to downloaded model file
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{model_name}.pt"
    
    # Check if model already exists
    if model_path.exists() and not force_reload:
        print(f"✓ Model already exists at: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        return str(model_path)
    
    print(f"Downloading {model_name} model...")
    print(f"This may take a few minutes depending on your internet speed.\n")
    
    # Download using torch hub (automatic)
    try:
        # This will download the model from ultralytics/yolov5 repository
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"  Path: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        return str(model_path)
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nTrying alternative download method...")
        
        # Alternative: Direct download from GitHub releases
        base_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0"
        model_url = f"{base_url}/{model_name}.pt"
        
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc=f"Downloading {model_name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"  Path: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        return str(model_path)


def verify_model(model_path: str) -> None:
    """
    Verify that the downloaded model can be loaded
    
    Args:
        model_path: Path to model file
    """
    print("\nVerifying model...")
    
    try:
        # Load model with YOLOv5
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        'Re-instantiates the architecture and maps the saved .pt state dictionary into memory to confirm file integrity.'
        # Get model info
        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Number of classes: {model.names.__len__()}")
        if hasattr(model, 'stride'):
            stride_val = max(model.stride) if isinstance(model.stride, (list, tuple)) else model.stride
            if hasattr(stride_val, 'item'):  # If it's a PyTorch Tensor
                stride_val = stride_val.item()
            print(f"  Input size: {int(stride_val * 8)}")
        else:
            print("  Input size: Variable")
        
        # Test inference on dummy input
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        'Generates synthetic standard image data (640x640 resolution, 3 channels - RGB, 8-bit unsigned integer values $[0, 255]$).'
        print("\nTesting inference on dummy image...")
        results = model(dummy_image)
        'Runs a standard forward-pass inference to ensure layer dimensions, strides, and anchors pass basic functional validation without crashing.'
        print(f"✓ Inference test successful!")
        print(f"  Detected {len(results.xyxy[0])} objects")
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        sys.exit(1)


def print_model_info():
    """Print information about available YOLOv5 models"""
    
    info = """
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    YOLOv5 Model Variants                           ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║ Model    │ Size (MB) │ mAP@0.5:0.95 │ Speed (ms) │ Params (M)    ║
    ╠══════════╪═══════════╪══════════════╪════════════╪════════════════╣
    ║ YOLOv5n  │    3.9    │    28.0%     │    4.5     │     1.9       ║
    ║ YOLOv5s  │   14.4    │    37.4%     │    6.4     │     7.2       ║
    ║ YOLOv5m  │   42.2    │    45.4%     │   12.3     │    21.2       ║
    ║ YOLOv5l  │   91.0    │    49.0%     │   20.9     │    46.5       ║
    ║ YOLOv5x  │  173.1    │    50.7%     │   38.3     │    86.7       ║
    ╚══════════╧═══════════╧══════════════╧════════════╧════════════════╝
    
    For this project, we use YOLOv5s as it offers:
    ✓ Good balance between speed and accuracy
    ✓ Manageable size for quantization experiments
    ✓ Reasonable inference time on CPU
    """
    print(info)


def main():
    """Main function"""
    print("="*70)
    print("YOLOv5 Model Download Script")
    print("="*70)
    print("="*70 + "\n")
    
    # Print model information
    print_model_info()
    
    # Configuration
    MODEL_NAME = 'yolov5s'  # Using small variant for this project
    OUTPUT_DIR = 'models'
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print()
    
    # Download model
    try:
        model_path = download_yolov5_model(
            model_name=MODEL_NAME,
            output_dir=OUTPUT_DIR,
            force_reload=False
        )
        
        # Verify model
        verify_model(model_path)
        
        print("\n" + "="*70)
        print("SUCCESS! Model is ready for conversion.")
        print("="*70)
        print(f"\nNext step: Run script 2 to convert PyTorch model to TensorFlow")
        print(f"  $ python scripts/2_convert_to_tensorflow.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

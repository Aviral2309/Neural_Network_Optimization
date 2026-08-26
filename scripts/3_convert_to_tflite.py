#!/usr/bin/env python3
"""
Script 3: Convert TensorFlow to TensorFlow Lite with Quantization

This is the CORE OPTIMIZATION script that applies quantization techniques:
- FP32: Full precision (baseline)
- FP16: Half precision (2x compression)
- INT8: Integer quantization (4x compression with calibration)

Quantization reduces model size and improves inference speed by:
1. Reducing numerical precision (32-bit → 16-bit → 8-bit)
2. Using integer arithmetic instead of floating-point
3. Enabling specialized CPU instructions (SIMD)

"""

import os
import sys
from pathlib import Path
from typing import List, Callable
import numpy as np
import tensorflow as tf

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset import (
    download_coco_subset,
    create_representative_dataset,
    representative_dataset_generator
    
)
'''
project ke parent direcotry ko python path me insert kar rha hai
taki internal modules import ho sakein.

representative_dataset_generator - int8 calibration ke liye data generator import kar rhe hai
'''

def convert_to_fp32_tflite(
    saved_model_dir: str,
    output_path: str
) -> str:
    """
    Convert TensorFlow SavedModel to FP32 TFLite (baseline)
    
    FP32 (32-bit floating point):
    - No quantization applied
    - Full precision maintained
    - Largest model size
    - Baseline for comparison
    
    Args:
        saved_model_dir: Path to TensorFlow SavedModel
        output_path: Output path for .tflite file
    
    Returns:
        Path to generated .tflite file
    """
    print("\n" + "="*70)
    print("Converting to FP32 TFLite (Baseline)")
    print("="*70)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # No optimization - full precision
    # This is the baseline model
    
    # Convert
    print("Converting... (this may take a few minutes)")
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model info
    model_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ FP32 conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {model_size:.2f} MB")
    print(f"  Precision: 32-bit float")
    print(f"  Optimization: None (baseline)")
    
    return str(output_path)


def convert_to_fp16_tflite(
    saved_model_dir: str,
    output_path: str
) -> str:
    """
    Convert TensorFlow SavedModel to FP16 TFLite
    
    FP16 (16-bit floating point):
    - Half precision quantization
    - ~2x size reduction
    - Minimal accuracy loss
    - Good for GPU/modern CPU
    
    How it works:
    - Weights converted from FP32 to FP16
    - Activations can remain FP32 or FP16
    - Reduces memory bandwidth requirements
    
    Args:
        saved_model_dir: Path to TensorFlow SavedModel
        output_path: Output path for .tflite file
    
    Returns:
        Path to generated .tflite file
    """
    print("\n" + "="*70)
    print("Converting to FP16 TFLite (Half Precision)")
    print("="*70)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enable FP16 quantization
    # This reduces all FP32 weights and activations to FP16
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    print("Converting with FP16 optimization...")
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model info
    model_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ FP16 conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {model_size:.2f} MB")
    print(f"  Precision: 16-bit float")
    print(f"  Optimization: Weight quantization to FP16")
    print(f"  Expected: ~2x size reduction vs FP32")
    
    return str(output_path)


def convert_to_int8_tflite(
    saved_model_dir: str,
    output_path: str,
    representative_dataset_fn: Callable
) -> str:
    """
    Convert TensorFlow SavedModel to INT8 TFLite (full integer quantization)
    
    INT8 (8-bit integer):
    - Full integer quantization
    - ~4x size reduction
    - Significant speedup on CPU
    - Requires representative dataset for calibration
    
    How it works:
    1. Calibration: Run representative inputs through model
    2. Statistics: Collect min/max values for each layer
    3. Quantization: Calculate scale and zero-point for each layer
    4. Conversion: Convert weights and activations to INT8
    
    Quantization formula:
        real_value = (quantized_value - zero_point) * scale
        quantized_value = round(real_value / scale) + zero_point
    
    Args:
        saved_model_dir: Path to TensorFlow SavedModel
        output_path: Output path for .tflite file
        representative_dataset_fn: Generator function yielding calibration data
    
    Returns:
        Path to generated .tflite file
    """
    print("\n" + "="*70)
    print("Converting to INT8 TFLite (Integer Quantization)")
    print("="*70)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enable INT8 quantization
    # This is the most aggressive optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for calibration
    # This is CRITICAL for INT8 quantization
    # The converter needs to see typical inputs to determine quantization parameters
    converter.representative_dataset = representative_dataset_fn
    
    # Enforce INT8 for all operations (full integer quantization)
    # Input and output will also be INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8
    
    # Convert
    print("Converting with INT8 optimization...")
    print("This requires calibration with representative dataset...")
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model info
    model_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ INT8 conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {model_size:.2f} MB")
    print(f"  Precision: 8-bit integer")
    print(f"  Optimization: Full integer quantization")
    print(f"  Expected: ~4x size reduction vs FP32")
    print(f"  Expected: ~4x speedup on CPU")
    
    return str(output_path)


def prepare_representative_dataset(
    num_samples: int = 200
) -> Callable:
    """
    Prepare representative dataset for INT8 quantization calibration
    
    Representative dataset requirements:
    - Should represent typical model inputs
    - 100-500 samples recommended
    - More samples = better calibration (but slower)
    - Should cover diverse input scenarios
    
    Args:
        num_samples: Number of calibration samples
    
    Returns:
        Generator function for representative dataset
    """
    print("\nPreparing representative dataset for INT8 calibration...")
    print(f"  Number of samples: {num_samples}")
    
    # Check if COCO data exists
    data_dir = Path("data/coco_val2017_subset")
    images_dir = data_dir / "images"
    
    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) < num_samples:
        print(f"\n  Downloading COCO validation subset...")
        image_paths, _ = download_coco_subset(
            output_dir=str(data_dir),
            num_images=num_samples,
            split='val2017'
        )
    else:
        print(f"  ✓ Using existing COCO images from: {images_dir}")
        image_paths = [str(p) for p in images_dir.glob("*.jpg")][:num_samples]
    
    # Create representative dataset generator
    print(f"  Creating generator for {len(image_paths)} images...")
    
    def representative_dataset():
        """Generator function for TFLite converter"""
        return representative_dataset_generator(
            image_paths=image_paths,
            input_size=(640, 640),
            max_samples=num_samples
        )
    
    print(f"  ✓ Representative dataset prepared")
    return representative_dataset


def verify_tflite_model(tflite_path: str):
    """
    Verify TFLite model by running test inference
    
    Args:
        tflite_path: Path to .tflite file
    """
    print(f"\nVerifying TFLite model: {Path(tflite_path).name}")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output type: {output_details[0]['dtype']}")
        
        # Create dummy input
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == np.uint8:
            dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        else:
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  ✓ Test inference successful")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    """Main function"""
    print("="*70)
    print("TensorFlow Lite Conversion with Quantization")
    print("="*70)
    print("="*70 + "\n")
    
    print("This script will create 3 optimized models:")
    print("  1. FP32 (baseline) - No quantization")
    print("  2. FP16 (2x smaller) - Half precision")
    print("  3. INT8 (4x smaller) - Integer quantization")
    print()
    
    # Configuration
    SAVED_MODEL_DIR = "models/yolov5s_saved_model"
    OUTPUT_DIR = "models"
    NUM_CALIBRATION_SAMPLES = 200  # For INT8 quantization
    
    # Check if SavedModel exists
    if not Path(SAVED_MODEL_DIR).exists():
        print(f"✗ TensorFlow SavedModel not found: {SAVED_MODEL_DIR}")
        print(f"\nPlease run script 2 first:")
        print(f"  $ python scripts/2_convert_to_tensorflow.py")
        sys.exit(1)
    
    print(f"Configuration:")
    print(f"  Input Model: {SAVED_MODEL_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Calibration Samples: {NUM_CALIBRATION_SAMPLES}")
    print()
    
    try:
        # 1. Convert to FP32 (baseline)
        fp32_path = convert_to_fp32_tflite(
            saved_model_dir=SAVED_MODEL_DIR,
            output_path=f"{OUTPUT_DIR}/yolov5s_fp32.tflite"
        )
        verify_tflite_model(fp32_path)
        
        # 2. Convert to FP16
        fp16_path = convert_to_fp16_tflite(
            saved_model_dir=SAVED_MODEL_DIR,
            output_path=f"{OUTPUT_DIR}/yolov5s_fp16.tflite"
        )
        verify_tflite_model(fp16_path)
        
        # 3. Prepare representative dataset for INT8
        representative_dataset = prepare_representative_dataset(
            num_samples=NUM_CALIBRATION_SAMPLES
        )
        
        # 4. Convert to INT8
        int8_path = convert_to_int8_tflite(
            saved_model_dir=SAVED_MODEL_DIR,
            output_path=f"{OUTPUT_DIR}/yolov5s_int8.tflite",
            representative_dataset_fn=representative_dataset
        )
        verify_tflite_model(int8_path)
        
        # Summary
        print("\n" + "="*70)
        print("SUCCESS! All TFLite models created successfully")
        print("="*70)
        
        print("\nModel Summary:")
        for model_path in [fp32_path, fp16_path, int8_path]:
            model_path = Path(model_path)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  {model_path.name:25s} {size_mb:>8.2f} MB")
        
        print("\nNext steps:")
        print("  1. Benchmark latency: python scripts/4_benchmark_latency.py")
        print("  2. Evaluate accuracy: python scripts/5_evaluate_accuracy.py")
        print("  3. Compare models: python scripts/6_compare_models.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

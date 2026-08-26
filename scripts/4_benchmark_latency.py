#!/usr/bin/env python3
"""
Script 4: Benchmark Inference Latency

This script measures inference latency for all TFLite model variants.
Proper benchmarking methodology includes:
- Warm-up runs to stabilize performance
- Multiple iterations for statistical significance
- CPU affinity and frequency scaling considerations
- Measuring percentiles (p50, p95, p99) not just mean

"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import calculate_inference_stats
from utils.dataset import preprocess_image, load_image


class TFLiteBenchmarker:
    """
    Benchmarker for TensorFlow Lite models
    
    This class provides accurate latency measurements with:
    - Warm-up runs
    - Statistical analysis
    - Memory profiling
    """
    
    def __init__(self, model_path: str, num_threads: int = 4):
        """
        Initialize benchmarker
        
        Args:
            model_path: Path to .tflite model
            num_threads: Number of CPU threads to use
        """
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        '''
        Yeh C++ level memory allocations invoke karta hai. 
        Model ke execution graph mein jetni static weights
        matrices aur dynamic activation tensors hain,
        unke liye memory memory-map (mmap) kar leta hai.
        '''
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"\nLoaded model: {self.model_name}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Input dtype: {self.input_dtype}")
    
    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare input for model
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed input tensor
        """
        # Preprocess image
        processed = preprocess_image(
            image,
            input_size=(self.input_shape[1], self.input_shape[2]),
            normalize=True
        )
        
        # Convert to correct dtype
        if self.input_dtype == np.uint8:
            # For INT8 models
            processed = (processed * 255).astype(np.uint8)
        else:
            # For FP32/FP16 models
            processed = processed.astype(np.float32)
        
        return processed
    
    def run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run single inference
        
        Args:
            input_tensor: Preprocessed input
        
        Returns:
            Model output
        """
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        """set_tensor(): Raw dynamic numpy memory block ko C++ memory pointer ke input tensor address par push karta hai.

invoke(): Main computation layer! C++ TFLite engine execute hota hai, jisme GEMM (General Matrix Multiply) operations run hoti hain.

get_tensor(): Prediction tensors (bounding boxes, class scores, coordinates) extract karke Python environment ko deliver karta hai.
        """
        return output
    
    def warm_up(self, num_runs: int = 10) -> None:
        """
        Warm-up runs to stabilize performance
        
        Why warm-up is important:
        - First few runs are slower due to:
          * Cache misses
          * Thread pool initialization
          * Dynamic frequency scaling
        - Warm-up ensures consistent measurements
        
        Args:
            num_runs: Number of warm-up runs
            
        Yeh function 10 dummy random iterations run karta hai taaki 
        actual benchmarks CPU ke steady peak state par capture ho sakein.
        """
        print(f"\nWarming up ({num_runs} runs)...")
        
        # Create dummy input
        if self.input_dtype == np.uint8:
            dummy_input = np.random.randint(
                0, 255, self.input_shape, dtype=np.uint8
            )
        else:
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Run warm-up inferences
        for _ in range(num_runs):
            self.run_inference(dummy_input)
        
        print("  ✓ Warm-up complete")
    
    def benchmark(
        self,
        num_runs: int = 100,
        use_real_images: bool = True,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model latency
        
        Methodology:
        1. Warm-up: Stabilize performance
        2. Multiple runs: Statistical significance
        3. Time measurement: High-precision timer
        4. Statistics: Mean, median, percentiles
        
        Args:
            num_runs: Number of benchmark runs
            use_real_images: Use real images vs dummy data
            warmup_runs: Number of warm-up runs
        
        Returns:
            Dictionary with latency statistics
        """
        # Warm-up
        self.warm_up(warmup_runs)
        
        # Prepare inputs
        print(f"\nBenchmarking ({num_runs} runs)...")
        
        if use_real_images:
            # Load real images for more realistic benchmarks
            data_dir = Path("data/coco_val2017_subset/images")
            if data_dir.exists():
                image_paths = list(data_dir.glob("*.jpg"))[:num_runs]
                images = [load_image(str(p)) for p in tqdm(image_paths[:10], desc="Loading images")]
                
                # Repeat if we have fewer images than runs
                while len(images) < num_runs:
                    images.extend(images[:min(len(images), num_runs - len(images))])
                
                inputs = [self.prepare_input(img) for img in images[:num_runs]]
            else:
                use_real_images = False
        
        if not use_real_images:
            # Use dummy data
            print("  Using dummy data (real images not available)")
            if self.input_dtype == np.uint8:
                dummy_input = np.random.randint(
                    0, 255, self.input_shape, dtype=np.uint8
                )
            else:
                dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            inputs = [dummy_input] * num_runs
        
        # Run benchmark
        latencies = []
        
        for input_tensor in tqdm(inputs, desc="Benchmarking"):
            # Measure latency with high precision
            start_time = time.perf_counter()
            self.run_inference(input_tensor)
            end_time = time.perf_counter()
            
            # Convert to milliseconds
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        stats = calculate_inference_stats(latencies)
        
        # Add additional info
        stats['model_name'] = self.model_name
        stats['num_runs'] = num_runs
        stats['input_shape'] = self.input_shape.tolist()
        stats['input_dtype'] = str(self.input_dtype)
        
        return stats
    
    def get_model_size(self) -> float:
        """
        Get model file size in MB
        
        Returns:
            Model size in megabytes
        """
        size_bytes = Path(self.model_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return size_mb


def benchmark_all_models(
    model_dir: str = "models",
    num_runs: int = 100,
    output_file: str = "results/benchmark_results.json"
) -> Dict[str, Dict]:
    """
    Benchmark all TFLite models
    
    Args:
        model_dir: Directory containing .tflite models
        num_runs: Number of benchmark runs per model
        output_file: Path to save results
    
    Returns:
        Dictionary with benchmark results for all models
    """
    model_dir = Path(model_dir)
    
    # Find all TFLite models
    tflite_models = list(model_dir.glob("yolov5s_*.tflite"))
    
    if not tflite_models:
        print(f"✗ No TFLite models found in {model_dir}")
        print(f"\nPlease run script 3 first:")
        print(f"  $ python scripts/3_convert_to_tflite.py")
        sys.exit(1)
    
    print(f"Found {len(tflite_models)} models to benchmark:")
    for model_path in tflite_models:
        print(f"  - {model_path.name}")
    
    # Benchmark each model
    results = {}
    
    for model_path in tflite_models:
        print("\n" + "="*70)
        print(f"Benchmarking: {model_path.name}")
        print("="*70)
        
        # Create benchmarker
        benchmarker = TFLiteBenchmarker(
            model_path=str(model_path),
            num_threads=4  # Use 4 threads for CPU
        )
        
        # Run benchmark
        stats = benchmarker.benchmark(
            num_runs=num_runs,
            use_real_images=True,
            warmup_runs=10
        )
        
        # Add model size
        stats['size_mb'] = benchmarker.get_model_size()
        
        # Store results
        model_variant = model_path.stem.split('_')[-1].upper()  # FP32, FP16, INT8
        results[model_variant] = stats
        
        # Print summary
        print(f"\n📊 Results for {model_variant}:")
        print(f"  Model Size: {stats['size_mb']:.2f} MB")
        print(f"  Mean Latency: {stats['mean']:.2f} ms")
        print(f"  Median Latency: {stats['median']:.2f} ms")
        print(f"  Std Dev: {stats['std']:.2f} ms")
        print(f"  P95 Latency: {stats['p95']:.2f} ms")
        print(f"  P99 Latency: {stats['p99']:.2f} ms")
        print(f"  Min: {stats['min']:.2f} ms")
        print(f"  Max: {stats['max']:.2f} ms")
    
    # Calculate speedups relative to FP32
    if 'FP32' in results:
        baseline_latency = results['FP32']['mean']
        print("\n" + "="*70)
        print("Speedup Analysis (vs FP32 baseline)")
        print("="*70)
        
        for model_variant, stats in results.items():
            speedup = baseline_latency / stats['mean']
            size_reduction = results['FP32']['size_mb'] / stats['size_mb']
            
            print(f"\n{model_variant}:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Size Reduction: {size_reduction:.2f}x")
            print(f"  Latency: {stats['mean']:.2f} ms (vs {baseline_latency:.2f} ms)")
            
            # Add to results
            results[model_variant]['speedup_vs_fp32'] = speedup
            results[model_variant]['size_reduction_vs_fp32'] = size_reduction
    
    # Save results
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Also save as CSV for easy viewing
    import pandas as pd
    
    csv_data = []
    for model_variant, stats in results.items():
        csv_data.append({
            'Model': model_variant,
            'Size (MB)': f"{stats['size_mb']:.2f}",
            'Mean Latency (ms)': f"{stats['mean']:.2f}",
            'Median (ms)': f"{stats['median']:.2f}",
            'Std Dev (ms)': f"{stats['std']:.2f}",
            'P95 (ms)': f"{stats['p95']:.2f}",
            'P99 (ms)': f"{stats['p99']:.2f}",
            'Speedup': f"{stats.get('speedup_vs_fp32', 1.0):.2f}x",
            'Size Reduction': f"{stats.get('size_reduction_vs_fp32', 1.0):.2f}x"
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = output_file.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"✓ CSV results saved to: {csv_path}")
    print("\n" + df.to_string(index=False))
    
    return results


def main():
    """Main function"""
    print("="*70)
    print("TFLite Model Latency Benchmarking")
    print("="*70)
    print("="*70 + "\n")
    
    # Configuration
    MODEL_DIR = "models"
    NUM_RUNS = 100  # More runs = better statistical significance
    OUTPUT_FILE = "results/benchmark_results.json"
    
    print(f"Configuration:")
    print(f"  Model Directory: {MODEL_DIR}")
    print(f"  Number of Runs: {NUM_RUNS}")
    print(f"  Output File: {OUTPUT_FILE}")
    print()
    
    print("Benchmarking Methodology:")
    print(f"  1. Warm-up runs: 10 (to stabilize performance)")
    print(f"  2. Benchmark runs: {NUM_RUNS}")
    print(f"  3. Statistical measures: mean, median, std, p95, p99")
    print(f"  4. Timer: High-precision (time.perf_counter)")
    print()
    
    try:
        results = benchmark_all_models(
            model_dir=MODEL_DIR,
            num_runs=NUM_RUNS,
            output_file=OUTPUT_FILE
        )
        
        print("\n" + "="*70)
        print("SUCCESS! Benchmarking completed")
        print("="*70)
        print(f"\nResults saved to:")
        print(f"  JSON: {OUTPUT_FILE}")
        print(f"  CSV: {Path(OUTPUT_FILE).with_suffix('.csv')}")
        
        print("\nNext steps:")
        print("  - Evaluate accuracy: python scripts/5_evaluate_accuracy.py")
        print("  - Compare all metrics: python scripts/6_compare_models.py")
        print("  - Visualize results: python scripts/7_visualize_results.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

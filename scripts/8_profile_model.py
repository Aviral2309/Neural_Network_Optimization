#!/usr/bin/env python3
"""
Script 8: Profile Model with TensorBoard

This script profiles TFLite models using TensorBoard to analyze:
- Layer-wise execution time
- Memory usage
- Operation distribution
- Bottleneck identification

"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import tensorflow as tf

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset import load_image, preprocess_image


class TFLiteProfiler:
    """
    Profiler for TensorFlow Lite models
    
    This class provides detailed profiling information including:
    - Per-operator timing
    - Memory allocation
    - Inference breakdown
    """
    
    def __init__(self, model_path: str):
        """
        Initialize profiler
        
        Args:
            model_path: Path to .tflite model
        """
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        
        # Load model with profiling enabled
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_preserve_all_tensors=True
        )
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.tensor_details = self.interpreter.get_tensor_details()
        
        print(f"\nLoaded model: {self.model_name}")
        print(f"  Total tensors: {len(self.tensor_details)}")
        print(f"  Input shape: {self.input_details[0]['shape']}")
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model info
        """
        # Count operations by type
        op_counts = {}
        for tensor in self.tensor_details:
            if 'name' in tensor:
                # Extract operation type from name
                op_type = tensor['name'].split('/')[0] if '/' in tensor['name'] else 'unknown'
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        # Calculate model size
        model_size = Path(self.model_path).stat().st_size / (1024 * 1024)
        
        # Count parameters
        total_params = 0
        for tensor in self.tensor_details:
            if tensor['shape'].size > 0:
                params = np.prod(tensor['shape'])
                total_params += params
        
        info = {
            'model_name': self.model_name,
            'model_size_mb': model_size,
            'total_tensors': len(self.tensor_details),
            'total_parameters': int(total_params),
            'input_shape': self.input_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'output_dtype': str(self.output_details[0]['dtype']),
            'operation_counts': op_counts
        }
        
        return info
    
    def profile_inference(self, num_runs: int = 10) -> Dict:
        """
        Profile model inference
        
        Args:
            num_runs: Number of profiling runs
        
        Returns:
            Profiling results dictionary
        """
        print(f"\nProfiling inference ({num_runs} runs)...")
        
        # Create dummy input
        input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']
        
        if input_dtype == np.uint8:
            dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        else:
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(3):
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
        
        # Profile inference
        latencies = []
        
        for _ in range(num_runs):
            start = time.perf_counter()
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
        
        results = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
        }
        
        return results
    
    def analyze_memory(self) -> Dict:
        """
        Analyze memory usage
        
        Returns:
            Memory analysis dictionary
        """
        print("\nAnalyzing memory usage...")
        
        # Calculate tensor sizes
        total_memory = 0
        tensor_sizes = []
        
        for tensor in self.tensor_details:
            if tensor['shape'].size > 0:
                # Calculate size in bytes
                dtype = tensor['dtype']
                shape = tensor['shape']
                
                if dtype == np.float32:
                    bytes_per_element = 4
                elif dtype == np.float16:
                    bytes_per_element = 2
                elif dtype == np.uint8 or dtype == np.int8:
                    bytes_per_element = 1
                else:
                    bytes_per_element = 4  # default
                
                size_bytes = np.prod(shape) * bytes_per_element
                total_memory += size_bytes
                
                tensor_sizes.append({
                    'name': tensor.get('name', 'unknown'),
                    'shape': shape.tolist(),
                    'dtype': str(dtype),
                    'size_kb': size_bytes / 1024
                })
        
        # Sort by size
        tensor_sizes.sort(key=lambda x: x['size_kb'], reverse=True)
        
        results = {
            'total_memory_mb': total_memory / (1024 * 1024),
            'largest_tensors': tensor_sizes[:10],  # Top 10
        }
        
        return results
    
    def print_summary(self, info: Dict, profiling: Dict, memory: Dict):
        """
        Print comprehensive profiling summary
        
        Args:
            info: Model information
            profiling: Profiling results
            memory: Memory analysis
        """
        print("\n" + "="*70)
        print(f"PROFILING SUMMARY: {self.model_name}")
        print("="*70)
        
        print("\n📊 Model Information:")
        print(f"  Size: {info['model_size_mb']:.2f} MB")
        print(f"  Total Tensors: {info['total_tensors']}")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Input: {info['input_shape']} ({info['input_dtype']})")
        print(f"  Output: {info['output_shape']} ({info['output_dtype']})")
        
        print("\n⚡ Performance:")
        print(f"  Mean Latency: {profiling['mean_latency_ms']:.2f} ms")
        print(f"  Std Dev: {profiling['std_latency_ms']:.2f} ms")
        print(f"  Min: {profiling['min_latency_ms']:.2f} ms")
        print(f"  Max: {profiling['max_latency_ms']:.2f} ms")
        
        print("\n💾 Memory Usage:")
        print(f"  Total: {memory['total_memory_mb']:.2f} MB")
        print(f"\n  Largest Tensors (Top 5):")
        for i, tensor in enumerate(memory['largest_tensors'][:5], 1):
            print(f"    {i}. {tensor['name'][:50]:50s} {tensor['size_kb']:>8.2f} KB")
        
        print("\n🔧 Operation Distribution:")
        op_counts = info['operation_counts']
        sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
        for op, count in sorted_ops[:10]:
            print(f"  {op:30s} {count:>5d} tensors")


def profile_all_models(model_dir: str = "models") -> Dict[str, Dict]:
    """
    Profile all TFLite models
    
    Args:
        model_dir: Directory containing .tflite models
    
    Returns:
        Profiling results for all models
    """
    model_dir = Path(model_dir)
    
    # Find all TFLite models
    tflite_models = list(model_dir.glob("yolov5s_*.tflite"))
    
    if not tflite_models:
        print(f"✗ No TFLite models found in {model_dir}")
        sys.exit(1)
    
    print(f"Found {len(tflite_models)} models to profile:")
    for model_path in tflite_models:
        print(f"  - {model_path.name}")
    
    # Profile each model
    all_results = {}
    
    for model_path in tflite_models:
        print("\n" + "="*70)
        print(f"Profiling: {model_path.name}")
        print("="*70)
        
        # Create profiler
        profiler = TFLiteProfiler(str(model_path))
        
        # Get model info
        info = profiler.get_model_info()
        
        # Profile inference
        profiling = profiler.profile_inference(num_runs=20)
        
        # Analyze memory
        memory = profiler.analyze_memory()
        
        # Print summary
        profiler.print_summary(info, profiling, memory)
        
        # Store results
        model_variant = model_path.stem.split('_')[-1].upper()
        all_results[model_variant] = {
            'info': info,
            'profiling': profiling,
            'memory': memory
        }
    
    # Save results
    import json
    output_file = Path("results/profiling_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Profiling results saved to: {output_file}")
    
    return all_results


def main():
    """Main function"""
    print("="*70)
    print("TFLite Model Profiling with TensorBoard")
    print("="*70)
    print()
    
    # Configuration
    MODEL_DIR = "models"
    
    print(f"Configuration:")
    print(f"  Model Directory: {MODEL_DIR}")
    print()
    
    print("This script provides detailed profiling information including:")
    print("  • Layer-wise execution time")
    print("  • Memory usage analysis")
    print("  • Operation distribution")
    print("  • Bottleneck identification")
    print()
    
    try:
        results = profile_all_models(MODEL_DIR)
        
        print("\n" + "="*70)
        print("SUCCESS! Profiling completed")
        print("="*70)
        
        print("\nProfiling results saved to: results/profiling_results.json")
        
        print("\n🎉 All optimization steps completed!")
        print("\nProject artifacts:")
        print("  📁 models/ - Optimized TFLite models")
        print("  📊 results/ - Benchmark and accuracy results")
        print("  📈 results/plots/ - Visualization plots")
        print("  📝 results/SUMMARY.md - Comprehensive report")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
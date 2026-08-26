#!/usr/bin/env python3
"""
Script 6: Compare All Model Variants

This script combines latency and accuracy results to provide a comprehensive
comparison of all quantization strategies. It generates:
- Comparison table (CSV)
- Summary statistics
- Tradeoff analysis

"""

import os
import sys
import json
from pathlib import Path
from typing import Dict
import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualization import create_comparison_table


def load_results() -> tuple:
    """
    Load benchmark and accuracy results
    
    Returns:
        Tuple of (benchmark_results, accuracy_results)
    """
    benchmark_file = Path("results/benchmark_results.json")
    accuracy_file = Path("results/accuracy_results.json")
    
    if not benchmark_file.exists():
        print(f"✗ Benchmark results not found: {benchmark_file}")
        print(f"\nPlease run script 4 first:")
        print(f"  $ python scripts/4_benchmark_latency.py")
        sys.exit(1)
    
    if not accuracy_file.exists():
        print(f"✗ Accuracy results not found: {accuracy_file}")
        print(f"\nPlease run script 5 first:")
        print(f"  $ python scripts/5_evaluate_accuracy.py")
        sys.exit(1)
    
    with open(benchmark_file, 'r') as f:
        benchmark_results = json.load(f)
    
    with open(accuracy_file, 'r') as f:
        accuracy_results = json.load(f)
    
    return benchmark_results, accuracy_results


def merge_results(
    benchmark_results: Dict,
    accuracy_results: Dict
) -> Dict[str, Dict]:
    """
    Merge benchmark and accuracy results
    
    Args:
        benchmark_results: Latency benchmark results
        accuracy_results: Accuracy evaluation results
    
    Returns:
        Merged results dictionary
    """
    merged = {}
    
    # Get all model variants
    models = set(benchmark_results.keys()) | set(accuracy_results.keys())
    
    for model in models:
        merged[model] = {}
        
        # Add benchmark metrics
        if model in benchmark_results:
            bench = benchmark_results[model]
            merged[model].update({
                'size_mb': bench.get('size_mb', 0),
                'latency_mean': bench.get('mean', 0),
                'latency_std': bench.get('std', 0),
                'latency_median': bench.get('median', 0),
                'latency_p95': bench.get('p95', 0),
                'latency_p99': bench.get('p99', 0),
                'speedup_vs_fp32': bench.get('speedup_vs_fp32', 1.0),
                'size_reduction_vs_fp32': bench.get('size_reduction_vs_fp32', 1.0),
            })
        
        # Add accuracy metrics
        if model in accuracy_results:
            acc = accuracy_results[model]
            merged[model].update({
                'mAP@0.50': acc.get('mAP@0.50', 0),
                'mAP@0.75': acc.get('mAP@0.75', 0),
                'mAP@0.5:0.95': acc.get('mAP@0.5:0.95', 0),
                'accuracy_drop_vs_fp32': acc.get('accuracy_drop_vs_fp32', 0),
            })
    
    return merged


def generate_comparison_report(results: Dict[str, Dict]) -> None:
    """
    Generate comprehensive comparison report
    
    Args:
        results: Merged results dictionary
    """
    print("\n" + "="*80)
    print(" "*25 + "MODEL COMPARISON REPORT")
    print("="*80 + "\n")
    
    # Sort models by precision (FP32, FP16, INT8)
    model_order = ['FP32', 'FP16', 'INT8']
    models = [m for m in model_order if m in results]
    
    # Print header
    print(f"{'Model':<10} {'Size':>10} {'Latency':>12} {'Speedup':>10} {'mAP@0.5:0.95':>15} {'Accuracy':>12}")
    print(f"{'':<10} {'(MB)':>10} {'(ms)':>12} {'(vs FP32)':>10} {'(%)':>15} {'Drop (%)':>12}")
    print("-" * 80)
    
    # Print each model
    for model in models:
        data = results[model]
        
        size = data.get('size_mb', 0)
        latency = data.get('latency_mean', 0)
        speedup = data.get('speedup_vs_fp32', 1.0)
        map_score = data.get('mAP@0.5:0.95', 0) * 100
        acc_drop = data.get('accuracy_drop_vs_fp32', 0)
        
        print(f"{model:<10} {size:>10.2f} {latency:>12.2f} {speedup:>9.2f}x {map_score:>14.2f} {acc_drop:>11.2f}")
    
    print("-" * 80)
    
    # Print insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80 + "\n")
    
    if 'FP32' in results and 'INT8' in results:
        fp32 = results['FP32']
        int8 = results['INT8']
        
        speedup = int8.get('speedup_vs_fp32', 1.0)
        size_reduction = int8.get('size_reduction_vs_fp32', 1.0)
        acc_drop = int8.get('accuracy_drop_vs_fp32', 0)
        
        print("INT8 Quantization Impact:")
        print(f"  ✓ {speedup:.2f}x faster inference")
        print(f"  ✓ {size_reduction:.2f}x smaller model")
        print(f"  ✓ Only {acc_drop:.2f}% accuracy drop")
        print()
        
        # Efficiency score (speedup * size_reduction / accuracy_drop)
        if acc_drop > 0:
            efficiency = (speedup * size_reduction) / (1 + acc_drop)
            print(f"  Efficiency Score: {efficiency:.2f}")
            print(f"  (Higher is better: speedup × compression / accuracy_loss)")
        print()
    
    if 'FP16' in results:
        fp16 = results['FP16']
        speedup_fp16 = fp16.get('speedup_vs_fp32', 1.0)
        size_reduction_fp16 = fp16.get('size_reduction_vs_fp32', 1.0)
        acc_drop_fp16 = fp16.get('accuracy_drop_vs_fp32', 0)
        
        print("FP16 Quantization Impact:")
        print(f"  ✓ {speedup_fp16:.2f}x faster inference")
        print(f"  ✓ {size_reduction_fp16:.2f}x smaller model")
        print(f"  ✓ Only {acc_drop_fp16:.2f}% accuracy drop")
        print()
    
    # Recommendations
    print("="*80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("="*80 + "\n")
    
    print("🎯 INT8 Quantization:")
    print("  • Best for: Edge devices, mobile, real-time applications")
    print("  • Pros: Maximum speedup and compression")
    print("  • Cons: Slight accuracy loss, requires calibration")
    print()
    
    print("⚖️  FP16 Quantization:")
    print("  • Best for: GPU deployment, balance of speed and accuracy")
    print("  • Pros: Good compression, minimal accuracy loss")
    print("  • Cons: Less speedup than INT8 on CPU")
    print()
    
    print("📦 FP32 (No Quantization):")
    print("  • Best for: Maximum accuracy requirements, abundant compute")
    print("  • Pros: Best accuracy, no calibration needed")
    print("  • Cons: Largest size, slowest inference")
    print()


def save_combined_results(results: Dict[str, Dict], output_dir: str = "results") -> None:
    """
    Save combined results to JSON and CSV
    
    Args:
        results: Merged results dictionary
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "combined_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Combined results saved to: {json_path}")
    
    # Create detailed CSV
    csv_data = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Size (MB)': f"{metrics.get('size_mb', 0):.2f}",
            'Latency Mean (ms)': f"{metrics.get('latency_mean', 0):.2f}",
            'Latency Std (ms)': f"{metrics.get('latency_std', 0):.2f}",
            'Latency P95 (ms)': f"{metrics.get('latency_p95', 0):.2f}",
            'Speedup vs FP32': f"{metrics.get('speedup_vs_fp32', 1.0):.2f}x",
            'Size Reduction vs FP32': f"{metrics.get('size_reduction_vs_fp32', 1.0):.2f}x",
            'mAP@0.50 (%)': f"{metrics.get('mAP@0.50', 0) * 100:.2f}",
            'mAP@0.75 (%)': f"{metrics.get('mAP@0.75', 0) * 100:.2f}",
            'mAP@0.5:0.95 (%)': f"{metrics.get('mAP@0.5:0.95', 0) * 100:.2f}",
            'Accuracy Drop (%)': f"{metrics.get('accuracy_drop_vs_fp32', 0):.2f}",
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "combined_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV results saved to: {csv_path}")
    
    # Print table
    print("\n" + "="*80)
    print("DETAILED COMPARISON TABLE")
    print("="*80 + "\n")
    print(df.to_string(index=False))


def main():
    """Main function"""
    print("="*80)
    print("Model Comparison Script")
    print("="*80)
    print("="*80 + "\n")
    
    try:
        # Load results
        print("Loading results...")
        benchmark_results, accuracy_results = load_results()
        print("  ✓ Benchmark results loaded")
        print("  ✓ Accuracy results loaded")
        
        # Merge results
        print("\nMerging results...")
        combined_results = merge_results(benchmark_results, accuracy_results)
        print(f"  ✓ Combined data for {len(combined_results)} models")
        
        # Generate report
        generate_comparison_report(combined_results)
        
        # Save combined results
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80 + "\n")
        save_combined_results(combined_results)
        
        print("\n" + "="*80)
        print("SUCCESS! Comparison completed")
        print("="*80)
        
        print("\nNext step:")
        print("  - Visualize results: python scripts/7_visualize_results.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

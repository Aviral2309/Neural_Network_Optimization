"""
Visualization utilities for model comparison and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_latency_comparison(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot latency comparison across different model variants
    
    Args:
        results: Dictionary with model results {model_name: {metrics}}
        output_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = list(results.keys())
    latencies = [results[m]['latency_mean'] for m in models]
    latencies_std = [results[m]['latency_std'] for m in models]
    
    # Bar plot with error bars
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(models, latencies, yerr=latencies_std, 
                   capsize=5, color=colors, alpha=0.7)
    
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{lat:.1f}ms',
                ha='center', va='bottom', fontweight='bold')
    
    # Speedup comparison (relative to FP32)
    baseline_latency = results['FP32']['latency_mean']
    speedups = [baseline_latency / results[m]['latency_mean'] for m in models]
    
    bars2 = ax2.bar(models, speedups, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax2.set_ylabel('Speedup (vs FP32)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Speedup', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add value labels
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved latency comparison plot to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_accuracy_comparison(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot accuracy comparison across different model variants
    
    Args:
        results: Dictionary with model results
        output_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = list(results.keys())
    map_50 = [results[m].get('mAP@0.50', 0) * 100 for m in models]
    map_50_95 = [results[m].get('mAP@0.5:0.95', 0) * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    colors = ['#3498db', '#e74c3c']
    bars1 = ax.bar(x - width/2, map_50, width, label='mAP@0.5', 
                   color=colors[0], alpha=0.7)
    bars2 = ax.bar(x + width/2, map_50_95, width, label='mAP@0.5:0.95',
                   color=colors[1], alpha=0.7)
    
    ax.set_xlabel('Model Variant', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison (COCO Metrics)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy comparison plot to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_size_comparison(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot model size comparison
    
    Args:
        results: Dictionary with model results
        output_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = list(results.keys())
    sizes_mb = [results[m]['size_mb'] for m in models]
    
    # Absolute size
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars1 = ax1.bar(models, sizes_mb, color=colors, alpha=0.7)
    
    ax1.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, size in zip(bars1, sizes_mb):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f}MB',
                ha='center', va='bottom', fontweight='bold')
    
    # Relative size (compression ratio)
    baseline_size = results['FP32']['size_mb']
    compression_ratios = [baseline_size / results[m]['size_mb'] for m in models]
    
    bars2 = ax2.bar(models, compression_ratios, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax2.set_ylabel('Compression Ratio (vs FP32)', fontsize=12, fontweight='bold')
    ax2.set_title('Size Reduction', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    for bar, ratio in zip(bars2, compression_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved size comparison plot to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_tradeoff_curve(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot latency vs accuracy tradeoff curve
    
    Args:
        results: Dictionary with model results
        output_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results.keys())
    latencies = [results[m]['latency_mean'] for m in models]
    accuracies = [results[m].get('mAP@0.5:0.95', 0) * 100 for m in models]
    sizes = [results[m]['size_mb'] for m in models]
    
    # Normalize sizes for marker size
    size_scale = [(s / max(sizes)) * 1000 + 100 for s in sizes]
    
    colors = {'FP32': '#3498db', 'FP16': '#2ecc71', 'INT8': '#e74c3c'}
    
    # Plot points
    for model, lat, acc, s in zip(models, latencies, accuracies, size_scale):
        ax.scatter(lat, acc, s=s, alpha=0.6, 
                  color=colors.get(model, '#95a5a6'),
                  edgecolors='black', linewidth=2,
                  label=model, zorder=5)
    
    # Connect points with line
    ax.plot(latencies, accuracies, 'k--', alpha=0.3, linewidth=2, zorder=1)
    
    # Annotate points
    for model, lat, acc in zip(models, latencies, accuracies):
        ax.annotate(model, 
                   xy=(lat, acc), 
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='yellow', 
                            alpha=0.3))
    
    # Add "Pareto frontier" region
    ax.fill_between([0, min(latencies)], 
                    [max(accuracies), max(accuracies)],
                    [max(accuracies), 0],
                    alpha=0.1, color='green',
                    label='Optimal Region')
    
    ax.set_xlabel('Inference Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('mAP@0.5:0.95 (%)', fontsize=13, fontweight='bold')
    ax.set_title('Latency vs Accuracy Tradeoff\n(Bubble size = Model size)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower left')
    
    # Set axis limits with padding
    ax.set_xlim(0, max(latencies) * 1.1)
    ax.set_ylim(min(accuracies) * 0.95, max(accuracies) * 1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved tradeoff curve to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_comparison_table(
    results: Dict[str, Dict],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a comparison table of all model variants
    
    Args:
        results: Dictionary with model results
        output_path: Path to save CSV file
    
    Returns:
        Pandas DataFrame with comparison
    """
    data = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Size (MB)': f"{metrics['size_mb']:.2f}",
            'Latency Mean (ms)': f"{metrics['latency_mean']:.2f}",
            'Latency Std (ms)': f"{metrics['latency_std']:.2f}",
            'Latency P95 (ms)': f"{metrics.get('latency_p95', 0):.2f}",
            'mAP@0.5 (%)': f"{metrics.get('mAP@0.50', 0) * 100:.2f}",
            'mAP@0.5:0.95 (%)': f"{metrics.get('mAP@0.5:0.95', 0) * 100:.2f}",
        }
        
        # Calculate relative metrics vs FP32
        if 'FP32' in results:
            baseline = results['FP32']
            row['Speedup vs FP32'] = f"{baseline['latency_mean'] / metrics['latency_mean']:.2f}x"
            row['Size Reduction vs FP32'] = f"{baseline['size_mb'] / metrics['size_mb']:.2f}x"
            acc_drop = (baseline.get('mAP@0.5:0.95', 0) - metrics.get('mAP@0.5:0.95', 0)) * 100
            row['Accuracy Drop (%)'] = f"{acc_drop:.2f}"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved comparison table to {output_path}")
    
    return df


def plot_latency_distribution(
    latencies_dict: Dict[str, List[float]],
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot latency distribution for each model variant
    
    Args:
        latencies_dict: Dictionary of {model_name: [latencies]}
        output_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(1, len(latencies_dict), figsize=(18, 5))
    
    if len(latencies_dict) == 1:
        axes = [axes]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for idx, (model_name, latencies) in enumerate(latencies_dict.items()):
        ax = axes[idx]
        
        # Histogram
        ax.hist(latencies, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
        
        # Add statistics
        mean = np.mean(latencies)
        median = np.median(latencies)
        
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}ms')
        ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}ms')
        
        ax.set_xlabel('Latency (ms)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} Latency Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved latency distribution plot to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test visualization functions with sample data
    print("Testing visualization utilities...")
    
    # Sample results
    sample_results = {
        'FP32': {
            'size_mb': 97.2,
            'latency_mean': 180.5,
            'latency_std': 5.3,
            'latency_p95': 190.2,
            'mAP@0.50': 0.768,
            'mAP@0.5:0.95': 0.542
        },
        'FP16': {
            'size_mb': 48.6,
            'latency_mean': 120.3,
            'latency_std': 3.8,
            'latency_p95': 126.5,
            'mAP@0.50': 0.767,
            'mAP@0.5:0.95': 0.541
        },
        'INT8': {
            'size_mb': 24.3,
            'latency_mean': 45.2,
            'latency_std': 2.1,
            'latency_p95': 48.5,
            'mAP@0.50': 0.754,
            'mAP@0.5:0.95': 0.528
        }
    }
    
    # Create test plots
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)
    
    plot_latency_comparison(sample_results, 
                           output_path=output_dir / "latency_comparison.png",
                           show_plot=False)
    
    plot_accuracy_comparison(sample_results,
                            output_path=output_dir / "accuracy_comparison.png",
                            show_plot=False)
    
    plot_size_comparison(sample_results,
                        output_path=output_dir / "size_comparison.png",
                        show_plot=False)
    
    plot_tradeoff_curve(sample_results,
                       output_path=output_dir / "tradeoff_curve.png",
                       show_plot=False)
    
    df = create_comparison_table(sample_results,
                                 output_path=output_dir / "comparison_table.csv")
    print("\nComparison Table:")
    print(df.to_string(index=False))
    
    print("\nVisualization utilities test completed!")
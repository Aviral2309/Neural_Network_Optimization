"""
Utility modules for YOLOv5 TensorFlow Lite Optimization

Author: Hitang Ketankumar Desai
Email: hitangdesai@gmail.com
GitHub: https://github.com/ihitangdesai
"""

from .dataset import (
    download_coco_subset,
    create_representative_dataset,
    load_image,
    preprocess_image
)
from .metrics import (
    calculate_map,
    calculate_iou,
    non_max_suppression
)
from .visualization import (
    plot_latency_comparison,
    plot_accuracy_comparison,
    plot_size_comparison,
    plot_tradeoff_curve,
    create_comparison_table
)

__version__ = "1.0.0"
__author__ = "Hitang Ketankumar Desai"

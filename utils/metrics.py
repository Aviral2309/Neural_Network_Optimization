"""
Metrics utilities for object detection evaluation

"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25
) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on detection boxes
    
    Args:
        boxes: Detection boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold
    
    Returns:
        Indices of boxes to keep
    """
    # Filter by score threshold
    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        # Keep the box with highest score
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = np.array([
            calculate_iou(current_box, box)
            for box in remaining_boxes
        ])
        
        # Keep boxes with IoU below threshold
        keep_mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return np.array(keep, dtype=np.int32)


def calculate_precision_recall(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_classes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_classes: List[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int = 80
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Calculate precision and recall for each class
    
    Args:
        pred_boxes: List of predicted boxes for each image [N, 4]
        pred_scores: List of prediction scores for each image [N]
        pred_classes: List of predicted classes for each image [N]
        gt_boxes: List of ground truth boxes for each image [M, 4]
        gt_classes: List of ground truth classes for each image [M]
        iou_threshold: IoU threshold for matching
        num_classes: Number of object classes
    
    Returns:
        Tuple of (precision dict, recall dict) per class
    """
    precisions = {}
    recalls = {}
    
    for class_id in range(num_classes):
        all_pred_scores = []
        all_matches = []
        num_gt = 0
        
        # Process each image
        for i in range(len(pred_boxes)):
            # Get predictions for this class
            class_mask = pred_classes[i] == class_id
            class_pred_boxes = pred_boxes[i][class_mask]
            class_pred_scores = pred_scores[i][class_mask]
            
            # Get ground truth for this class
            gt_mask = gt_classes[i] == class_id
            class_gt_boxes = gt_boxes[i][gt_mask]
            num_gt += len(class_gt_boxes)
            
            # Sort predictions by score
            if len(class_pred_scores) > 0:
                sorted_idx = np.argsort(class_pred_scores)[::-1]
                class_pred_boxes = class_pred_boxes[sorted_idx]
                class_pred_scores = class_pred_scores[sorted_idx]
            
            # Match predictions to ground truth
            matched_gt = set()
            for j, pred_box in enumerate(class_pred_boxes):
                all_pred_scores.append(class_pred_scores[j])
                
                best_iou = 0
                best_gt_idx = -1
                
                for k, gt_box in enumerate(class_gt_boxes):
                    if k in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = k
                
                # Check if match is above threshold
                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    all_matches.append(1)  # True positive
                    matched_gt.add(best_gt_idx)
                else:
                    all_matches.append(0)  # False positive
        
        if len(all_pred_scores) == 0:
            precisions[class_id] = np.array([])
            recalls[class_id] = np.array([])
            continue
        
        # Sort by confidence
        all_pred_scores = np.array(all_pred_scores)
        all_matches = np.array(all_matches)
        sorted_idx = np.argsort(all_pred_scores)[::-1]
        all_matches = all_matches[sorted_idx]
        
        # Calculate cumulative precision and recall
        tp_cumsum = np.cumsum(all_matches)
        fp_cumsum = np.cumsum(1 - all_matches)
        
        recalls_arr = tp_cumsum / max(num_gt, 1)
        precisions_arr = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        precisions[class_id] = precisions_arr
        recalls[class_id] = recalls_arr
    
    return precisions, recalls


def calculate_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using 11-point interpolation
    
    Args:
        precision: Precision values
        recall: Recall values
    
    Returns:
        Average Precision
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    # Add sentinel values
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    
    # Compute precision envelope
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # Calculate AP using 11-point interpolation
    recall_thresholds = np.linspace(0, 1, 11)
    ap = 0.0
    
    for threshold in recall_thresholds:
        # Find precision at this recall threshold
        indices = np.where(recall >= threshold)[0]
        if len(indices) > 0:
            ap += precision[indices[0]]
    
    return ap / 11.0


def calculate_map(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_classes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_classes: List[np.ndarray],
    iou_thresholds: List[float] = None,
    num_classes: int = 80
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) at different IoU thresholds
    
    Args:
        pred_boxes: Predicted boxes for each image
        pred_scores: Prediction confidence scores
        pred_classes: Predicted class labels
        gt_boxes: Ground truth boxes
        gt_classes: Ground truth class labels
        iou_thresholds: List of IoU thresholds (default: [0.5, 0.75, 0.5:0.95])
        num_classes: Number of object classes
    
    Returns:
        Dictionary with mAP metrics
    """
    if iou_thresholds is None:
        # Standard COCO evaluation thresholds
        iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))
    
    results = {}
    
    # Calculate mAP for each IoU threshold
    for iou_thresh in iou_thresholds:
        precisions, recalls = calculate_precision_recall(
            pred_boxes, pred_scores, pred_classes,
            gt_boxes, gt_classes,
            iou_threshold=iou_thresh,
            num_classes=num_classes
        )
        
        # Calculate AP for each class
        aps = []
        for class_id in range(num_classes):
            if len(precisions[class_id]) > 0:
                ap = calculate_ap(precisions[class_id], recalls[class_id])
                aps.append(ap)
        
        # Calculate mAP
        map_value = np.mean(aps) if aps else 0.0
        results[f'mAP@{iou_thresh:.2f}'] = map_value
    
    # Calculate mAP@0.5:0.95 (COCO standard)
    map_50_95 = np.mean([
        results[f'mAP@{thresh:.2f}']
        for thresh in np.arange(0.5, 1.0, 0.05)
    ])
    results['mAP@0.5:0.95'] = map_50_95
    
    return results


def calculate_inference_stats(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for inference latencies
    
    Args:
        latencies: List of latency measurements in milliseconds
    
    Returns:
        Dictionary with statistical metrics
    """
    latencies = np.array(latencies)
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p25': np.percentile(latencies, 25),
        'p50': np.percentile(latencies, 50),
        'p75': np.percentile(latencies, 75),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }


if __name__ == "__main__":
    # Test metrics functions
    print("Testing metrics utilities...")
    
    # Test IoU
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 5, 15, 15])
    iou = calculate_iou(box1, box2)
    print(f"IoU between boxes: {iou:.3f}")
    
    # Test NMS
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [20, 20, 30, 30]
    ])
    scores = np.array([0.9, 0.8, 0.95])
    keep = non_max_suppression(boxes, scores)
    print(f"NMS kept indices: {keep}")
    
    # Test inference stats
    latencies = [10.5, 11.2, 10.8, 12.1, 10.3]
    stats = calculate_inference_stats(latencies)
    print(f"Inference stats: mean={stats['mean']:.2f}ms, std={stats['std']:.2f}ms")
    
    print("Metrics utilities test completed!")

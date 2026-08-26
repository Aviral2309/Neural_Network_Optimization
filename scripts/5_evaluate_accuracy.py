#!/usr/bin/env python3
"""
Script 5: Evaluate Model Accuracy

This script evaluates the accuracy of quantized models on COCO validation set.
We measure mAP (mean Average Precision) to quantify the accuracy loss
from quantization.

Why accuracy evaluation is critical:
- Quantization trades accuracy for speed/size
- Need to measure this tradeoff quantitatively
- mAP is the standard metric for object detection

"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset import load_image, preprocess_image, download_coco_subset
from utils.metrics import non_max_suppression


class TFLiteEvaluator:
    """
    Evaluator for TFLite models on COCO dataset
    
    This class handles:
    - Loading COCO annotations
    - Running inference on validation images
    - Converting outputs to COCO format
    - Calculating mAP metrics
    """
    
    def __init__(self, model_path: str, coco_json: str):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to .tflite model
            coco_json: Path to COCO annotations JSON
        """
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        
        # Load COCO annotations
        print(f"\nLoading COCO annotations...")
        self.coco_gt = COCO(coco_json)
        self.image_ids = list(self.coco_gt.imgs.keys())
        
        print(f"  ✓ Loaded {len(self.image_ids)} images")
        
        # COCO class names (80 classes)
        self.class_names = [
            self.coco_gt.cats[cat_id]['name']
            for cat_id in sorted(self.coco_gt.cats.keys())
        ]
    
    def preprocess_input(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for model
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (preprocessed tensor, scale factor)
        """
        # Get original size
        h, w = image.shape[:2]
        
        # Preprocess
        processed = preprocess_image(
            image,
            input_size=(self.input_shape[1], self.input_shape[2]),
            normalize=True
        )
        
        # Convert to correct dtype
        if self.input_dtype == np.uint8:
            processed = (processed * 255).astype(np.uint8)
        else:
            processed = processed.astype(np.float32)
        
        # Calculate scale for bbox conversion
        scale = min(self.input_shape[1] / h, self.input_shape[2] / w)
        
        return processed, scale
    
    def postprocess_output(
        self,
        output: np.ndarray,
        scale: float,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess model output
        
        Args:
            output: Raw model output
            scale: Scale factor from preprocessing
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Tuple of (boxes, scores, classes)
        """
        # YOLOv5 output format: [batch, num_predictions, 85]
        # 85 = 4 (bbox) + 1 (objectness) + 80 (class scores)
        
        # For INT8 models, dequantize output
        if self.input_dtype == np.uint8:
            output_details = self.output_details[0]
            if 'quantization' in output_details:
                scale_out, zero_point = output_details['quantization']
                if scale_out != 0:
                    output = (output.astype(np.float32) - zero_point) * scale_out
        
        # Get predictions (remove batch dimension)
        predictions = output[0]
        
        # Filter by confidence
        obj_conf = predictions[:, 4]
        mask = obj_conf >= conf_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract components
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        class_confs = np.max(class_scores, axis=1)
        
        # Final confidence = objectness * class_confidence
        scores = obj_conf * class_confs
        
        # Convert boxes from center format to corner format
        # [x_center, y_center, w, h] -> [x1, y1, x2, y2]
        boxes_corner = np.zeros_like(boxes)
        boxes_corner[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_corner[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_corner[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_corner[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Scale boxes back to original image size
        boxes_corner = boxes_corner / scale
        
        # Apply NMS per class
        keep_indices = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes_corner[class_mask]
            class_scores = scores[class_mask]
            
            # Apply NMS
            keep = non_max_suppression(
                class_boxes,
                class_scores,
                iou_threshold=iou_threshold,
                score_threshold=conf_threshold
            )
            
            # Get original indices
            class_indices = np.where(class_mask)[0]
            keep_indices.extend(class_indices[keep])
        
        # Filter results
        if len(keep_indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        keep_indices = np.array(keep_indices)
        final_boxes = boxes_corner[keep_indices]
        final_scores = scores[keep_indices]
        final_classes = class_ids[keep_indices]
        
        return final_boxes, final_scores, final_classes
    
    def evaluate(
        self,
        num_images: int = None,
        save_detections: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on COCO validation set
        
        Args:
            num_images: Number of images to evaluate (None = all)
            save_detections: Whether to save detection results
        
        Returns:
            Dictionary with mAP metrics
        """
        if num_images:
            image_ids = self.image_ids[:num_images]
        else:
            image_ids = self.image_ids
        
        print(f"\nEvaluating on {len(image_ids)} images...")
        
        # Store detections in COCO format
        detections = []
        
        # Process each image
        for img_id in tqdm(image_ids, desc="Processing images"):
            # Load image
            img_info = self.coco_gt.imgs[img_id]
            img_path = Path("data/coco_val2017_subset/images") / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            image = load_image(str(img_path))
            
            # Preprocess
            input_tensor, scale = self.preprocess_input(image)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Postprocess
            boxes, scores, classes = self.postprocess_output(output, scale)
            
            # Convert to COCO format
            for box, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                
                detection = {
                    'image_id': int(img_id),
                    'category_id': int(class_id) + 1,  # COCO uses 1-indexed
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(score)
                }
                detections.append(detection)
        
        print(f"\n  Total detections: {len(detections)}")
        
        # Save detections if requested
        if save_detections:
            det_file = f"results/{self.model_name}_detections.json"
            Path(det_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(det_file, 'w') as f:
                json.dump(detections, f)
            
            print(f"  ✓ Detections saved to: {det_file}")
        
        # Evaluate using COCO API
        if len(detections) == 0:
            print("  ✗ No detections found!")
            return {
                'mAP@0.50': 0.0,
                'mAP@0.75': 0.0,
                'mAP@0.5:0.95': 0.0
            }
        
        print("\n  Computing COCO metrics...")
        
        # Load detections
        coco_dt = self.coco_gt.loadRes(detections)
        
        # Run evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        results = {
            'mAP@0.5:0.95': coco_eval.stats[0],  # AP @ IoU=0.50:0.95
            'mAP@0.50': coco_eval.stats[1],      # AP @ IoU=0.50
            'mAP@0.75': coco_eval.stats[2],      # AP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],     # AP for small objects
            'mAP_medium': coco_eval.stats[4],    # AP for medium objects
            'mAP_large': coco_eval.stats[5],     # AP for large objects
        }
        
        return results


def evaluate_all_models(
    model_dir: str = "models",
    coco_json: str = "data/coco_val2017_subset/annotations/instances_val2017.json",
    num_images: int = 1000,
    output_file: str = "results/accuracy_results.json"
) -> Dict[str, Dict]:
    """
    Evaluate all TFLite models
    
    Args:
        model_dir: Directory containing .tflite models
        coco_json: Path to COCO annotations
        num_images: Number of images to evaluate
        output_file: Path to save results
    
    Returns:
        Dictionary with accuracy results for all models
    """
    model_dir = Path(model_dir)
    
    # Check COCO data
    if not Path(coco_json).exists():
        print(f"COCO annotations not found. Downloading subset...")
        download_coco_subset(
            output_dir="data/coco_val2017_subset",
            num_images=num_images,
            split='val2017'
        )
    
    # Find all TFLite models
    tflite_models = list(model_dir.glob("yolov5s_*.tflite"))
    
    if not tflite_models:
        print(f"✗ No TFLite models found in {model_dir}")
        sys.exit(1)
    
    print(f"Found {len(tflite_models)} models to evaluate:")
    for model_path in tflite_models:
        print(f"  - {model_path.name}")
    
    # Evaluate each model
    results = {}
    
    for model_path in tflite_models:
        print("\n" + "="*70)
        print(f"Evaluating: {model_path.name}")
        print("="*70)
        
        # Create evaluator
        evaluator = TFLiteEvaluator(
            model_path=str(model_path),
            coco_json=coco_json
        )
        
        # Run evaluation
        metrics = evaluator.evaluate(
            num_images=num_images,
            save_detections=True
        )
        
        # Store results
        model_variant = model_path.stem.split('_')[-1].upper()
        results[model_variant] = metrics
        
        # Print summary
        print(f"\n📊 Results for {model_variant}:")
        print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f} ({metrics['mAP@0.5:0.95']*100:.2f}%)")
        print(f"  mAP@0.50: {metrics['mAP@0.50']:.4f} ({metrics['mAP@0.50']*100:.2f}%)")
        print(f"  mAP@0.75: {metrics['mAP@0.75']:.4f} ({metrics['mAP@0.75']*100:.2f}%)")
    
    # Calculate accuracy drop relative to FP32
    if 'FP32' in results:
        baseline_map = results['FP32']['mAP@0.5:0.95']
        
        print("\n" + "="*70)
        print("Accuracy Analysis (vs FP32 baseline)")
        print("="*70)
        
        for model_variant, metrics in results.items():
            accuracy_drop = (baseline_map - metrics['mAP@0.5:0.95']) * 100
            
            print(f"\n{model_variant}:")
            print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']*100:.2f}%")
            print(f"  Accuracy Drop: {accuracy_drop:.2f}% (vs {baseline_map*100:.2f}%)")
            
            # Add to results
            results[model_variant]['accuracy_drop_vs_fp32'] = accuracy_drop
    
    # Save results
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


def main():
    """Main function"""
    print("="*70)
    print("TFLite Model Accuracy Evaluation")
    print("="*70)
    print("="*70 + "\n")
    
    # Configuration
    MODEL_DIR = "models"
    COCO_JSON = "data/coco_val2017_subset/annotations/instances_val2017.json"
    NUM_IMAGES = 1000  # Evaluate on 1000 images for faster evaluation
    OUTPUT_FILE = "results/accuracy_results.json"
    
    print(f"Configuration:")
    print(f"  Model Directory: {MODEL_DIR}")
    print(f"  COCO Annotations: {COCO_JSON}")
    print(f"  Number of Images: {NUM_IMAGES}")
    print(f"  Output File: {OUTPUT_FILE}")
    print()
    
    print("Note: This evaluation may take 15-30 minutes depending on your CPU.")
    print("For faster testing, reduce NUM_IMAGES in the script.")
    print()
    
    try:
        results = evaluate_all_models(
            model_dir=MODEL_DIR,
            coco_json=COCO_JSON,
            num_images=NUM_IMAGES,
            output_file=OUTPUT_FILE
        )
        
        print("\n" + "="*70)
        print("SUCCESS! Accuracy evaluation completed")
        print("="*70)
        print(f"\nResults saved to: {OUTPUT_FILE}")
        
        print("\nNext steps:")
        print("  - Compare all metrics: python scripts/6_compare_models.py")
        print("  - Visualize results: python scripts/7_visualize_results.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

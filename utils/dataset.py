"""
Dataset utilities for YOLOv5 TensorFlow Lite optimization

"""

import os
import json
import requests
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """
    Download a file from URL with progress bar
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        chunk_size: Size of chunks to download at a time
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)


def download_coco_subset(
    output_dir: str,
    num_images: int = 1000,
    split: str = 'val2017'
) -> Tuple[List[str], str]:
    """
    Download a subset of COCO dataset for validation
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to download
        split: COCO split to use ('val2017' or 'train2017')
    
    Returns:
        Tuple of (list of image paths, annotations file path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    annotations_dir = output_dir / 'annotations'
    annotations_dir.mkdir(exist_ok=True)
    
    print(f"Downloading COCO {split} subset ({num_images} images)...")
    
    # Download annotations
    anno_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    anno_zip = annotations_dir / "annotations.zip"
    
    if not (annotations_dir / f"instances_{split}.json").exists():
        print("Downloading annotations...")
        download_file(anno_url, str(anno_zip))
        
        print("Extracting annotations...")
        with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        anno_zip.unlink()
    
    # Load annotations
    anno_file = annotations_dir / f"instances_{split}.json"
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    # Select subset of images
    images_to_download = coco_data['images'][:num_images]
    
    # Download images
    image_paths = []
    print(f"Downloading {len(images_to_download)} images...")
    
    for img_info in tqdm(images_to_download, desc="Downloading images"):
        img_filename = img_info['file_name']
        img_path = images_dir / img_filename
        
        if not img_path.exists():
            img_url = f"http://images.cocodataset.org/{split}/{img_filename}"
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    image_paths.append(str(img_path))
            except Exception as e:
                print(f"Failed to download {img_filename}: {e}")
        else:
            image_paths.append(str(img_path))
    
    print(f"Downloaded {len(image_paths)} images successfully")
    return image_paths, str(anno_file)


def create_representative_dataset(
    image_dir: str,
    num_samples: int = 200,
    input_size: Tuple[int, int] = (640, 640)
) -> List[str]:
    """
    Create a representative dataset for INT8 quantization calibration
    
    Args:
        image_dir: Directory containing images
        num_samples: Number of samples for calibration
        input_size: Input size for the model (height, width)
    
    Returns:
        List of image paths
    """
    image_dir = Path(image_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in image_dir.rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    # Select representative samples
    if len(image_files) > num_samples:
        # Use random sampling for diversity
        np.random.seed(42)
        indices = np.random.choice(len(image_files), num_samples, replace=False)
        selected_images = [image_files[i] for i in indices]
    else:
        selected_images = image_files
    
    print(f"Selected {len(selected_images)} images for representative dataset")
    return [str(img) for img in selected_images]


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (RGB)
    """
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def preprocess_image(
    image: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for YOLOv5 inference
    
    Args:
        image: Input image as numpy array
        input_size: Target size (height, width)
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Preprocessed image ready for inference
    """
    # Resize with aspect ratio preservation
    h, w = image.shape[:2]
    target_h, target_w = input_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    # Normalize if requested
    if normalize:
        padded = padded.astype(np.float32) / 255.0
    
    # Add batch dimension and return
    return np.expand_dims(padded, axis=0)


def representative_dataset_generator(
    image_paths: List[str],
    input_size: Tuple[int, int] = (640, 640),
    max_samples: Optional[int] = None
):
    """
    Generator function for TensorFlow Lite representative dataset
    
    This is used for INT8 quantization calibration. The generator yields
    preprocessed images that represent the typical input distribution.
    
    Args:
        image_paths: List of paths to calibration images
        input_size: Model input size (height, width)
        max_samples: Maximum number of samples to generate
    
    Yields:
        List containing preprocessed image batch
    """
    if max_samples:
        image_paths = image_paths[:max_samples]
    
    for image_path in tqdm(image_paths, desc="Generating representative dataset"):
        try:
            # Load and preprocess image
            image = load_image(image_path)
            processed = preprocess_image(image, input_size, normalize=True)
            
            # TFLite expects float32 input for calibration
            yield [processed.astype(np.float32)]
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue


def batch_generator(
    image_paths: List[str],
    batch_size: int = 1,
    input_size: Tuple[int, int] = (640, 640)
):
    """
    Generator for batch inference
    
    Args:
        image_paths: List of image paths
        batch_size: Batch size
        input_size: Model input size (height, width)
    
    Yields:
        Tuple of (preprocessed batch, original images)
    """
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        originals = []
        
        for path in batch_paths:
            try:
                img = load_image(path)
                processed = preprocess_image(img, input_size, normalize=True)
                images.append(processed)
                originals.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if images:
            yield np.vstack(images), originals


if __name__ == "__main__":
    # Test dataset utilities
    print("Testing dataset utilities...")
    
    # Create test directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Test image preprocessing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = preprocess_image(test_image, input_size=(640, 640))
    print(f"Preprocessed image shape: {processed.shape}")
    
    print("Dataset utilities test completed!")

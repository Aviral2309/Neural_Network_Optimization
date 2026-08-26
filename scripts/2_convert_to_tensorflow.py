"""
Script 2: Convert PyTorch Model to TensorFlow SavedModel Format
======================================================================
This script loads a PyTorch YOLOv5 weight file (.pt) and exports it into a 
TensorFlow SavedModel directory structure suitable for TFLite conversion.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root directory to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def check_dependencies():
    """Verify required packages are installed."""
    print("Checking dependencies...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not installed. Install with: pip install tensorflow")
        sys.exit(1)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed. Install with: pip install torch torchvision")
        sys.exit(1)


def convert_pytorch_to_tensorflow(
    pytorch_model_path: str,
    output_dir: str,
    input_size: int = 640
) -> str:
    """
    Convert PyTorch YOLOv5 model to TensorFlow SavedModel format.

    Args:
        pytorch_model_path: Path to the input PyTorch (.pt) file.
        output_dir: Base output directory for saved models.
        input_size: Image input dimension (square height/width).

    Returns:
        Path string to the created TensorFlow SavedModel directory.
    """
    pytorch_model_path = Path(pytorch_model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = output_dir / f"{pytorch_model_path.stem}_saved_model"

    # Fast return if the model has already been exported properly
    if saved_model_dir.exists() and (saved_model_dir / "saved_model.pb").exists():
        print(f"\n✓ TensorFlow SavedModel already exists at:\n  {saved_model_dir}")
        return str(saved_model_dir)

    print(f"\nConverting {pytorch_model_path} to TensorFlow...")
    print(f"Input size: {input_size}x{input_size}\n")

    # Clone YOLOv5 repository if not present locally
    yolov5_dir = project_root / "yolov5_repo"
    if not yolov5_dir.exists():
        print("Cloning YOLOv5 repository for conversion tools...")
        subprocess.run([
            "git", "clone", "https://github.com/ultralytics/yolov5.git",
            str(yolov5_dir)
        ], check=True)

    # Ensure repository is in Python search path
    if str(yolov5_dir) not in sys.path:
        sys.path.insert(0, str(yolov5_dir))

    try:
        print("Exporting model via Ultralytics engine...")
        import export

        # Run export in-process to avoid Windows subprocess output encoding bugs
        export.run(
            weights=str(pytorch_model_path),
            imgsz=(input_size, input_size),
            batch_size=1,
            include=['saved_model'],
            device='cpu'
        )

        # Validate output directory
        if saved_model_dir.exists() and (saved_model_dir / "saved_model.pb").exists():
            model_size = sum(
                f.stat().st_size for f in saved_model_dir.rglob('*') if f.is_file()
            ) / (1024 * 1024)

            print(f"\n✓ Conversion successful!")
            print(f"  TensorFlow SavedModel: {saved_model_dir}")
            print(f"  Model size: {model_size:.2f} MB")

            return str(saved_model_dir)
        else:
            raise FileNotFoundError(
                f"Export process completed, but missing 'saved_model.pb' in {saved_model_dir}"
            )

    except Exception as e:
        print(f"\n✗ Direct export failed: {e}")
        print("Attempting fallback using subprocess with explicit UTF-8 encoding...")

        export_script = yolov5_dir / "export.py"
        cmd = [
            sys.executable,
            str(export_script),
            "--weights", str(pytorch_model_path),
            "--include", "saved_model",
            "--imgsz", str(input_size),
            "--batch-size", "1"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            print("\nSubprocess output:")
            print(result.stdout)
            print("\nSubprocess errors:")
            print(result.stderr)
            raise RuntimeError("Subprocess conversion failed.")

        if saved_model_dir.exists() and (saved_model_dir / "saved_model.pb").exists():
            print(f"\n✓ Fallback conversion successful!")
            print(f"  TensorFlow SavedModel: {saved_model_dir}")
            return str(saved_model_dir)
        else:
            raise FileNotFoundError(f"Export failed. Directory not found: {saved_model_dir}")


def main():
    print("=" * 70)
    print("PyTorch to TensorFlow Conversion Script")
    print("=" * 70)

    check_dependencies()

    pytorch_model = project_root / "models" / "yolov5s.pt"
    output_dir = project_root / "models"
    input_size = 640

    print("\nConfiguration:")
    print(f"  PyTorch Model: {pytorch_model}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Input Size: {input_size}x{input_size}")

    if not pytorch_model.exists():
        print(f"\n✗ Error: Model file not found at {pytorch_model}")
        print("Please run script 1 to download the PyTorch model first.")
        sys.exit(1)

    try:
        tf_model_dir = convert_pytorch_to_tensorflow(
            pytorch_model_path=str(pytorch_model),
            output_dir=str(output_dir),
            input_size=input_size
        )
        print("\n" + "=" * 70)
        print("Conversion Complete!")
        print("=" * 70)
        print(f"Next step: Run script 3 to convert to TFLite and quantize:")
        print("  python scripts/3_convert_to_tflite.py")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
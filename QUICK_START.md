# Quick Start Guide

---

## 🚀 Getting Started in 5 Minutes

### Prerequisites

* Python 3.8+
* 4GB+ RAM
* 2GB+ free disk space
* Internet connection for downloading the model and dataset

### Installation

```bash
# 1. Clone or download the project
cd yolo-tflite-optimization

# 2. Run setup script
bash setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run complete pipeline
bash run_all.sh
```

That's it! The pipeline will automatically:

1. Download the YOLOv5 model
2. Convert the model to TensorFlow
3. Create FP32, FP16 and INT8 TFLite variants
4. Benchmark inference latency
5. Evaluate detection accuracy
6. Compare model variants
7. Generate visualizations
8. Generate the final results report

**Estimated time:** 30–60 minutes, depending on CPU performance and internet speed.

---

## 📂 Project Structure

```text
yolo-tflite-optimization/

├── README.md                  # Comprehensive documentation
├── QUICK_START.md             # This file
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
├── run_all.sh                 # Run complete pipeline
│
├── scripts/                   # Main scripts (run in order)
│   ├── 1_download_model.py
│   ├── 2_convert_to_tensorflow.py
│   ├── 3_convert_to_tflite.py
│   ├── 4_benchmark_latency.py
│   ├── 5_evaluate_accuracy.py
│   ├── 6_compare_models.py
│   ├── 7_visualize_results.py
│   └── 8_profile_model.py
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── dataset.py             # Dataset handling
│   ├── metrics.py             # Evaluation metrics
│   └── visualization.py       # Plotting functions
│
├── models/                    # Generated models
├── data/                      # Dataset and calibration data
└── results/                   # Results and visualizations
```

---

## 🎯 Quick Commands

### Run Individual Scripts

```bash
# Download model
python scripts/1_download_model.py

# Convert to TensorFlow
python scripts/2_convert_to_tensorflow.py

# Create FP32, FP16 and INT8 models
python scripts/3_convert_to_tflite.py

# Benchmark performance
python scripts/4_benchmark_latency.py

# Evaluate accuracy
python scripts/5_evaluate_accuracy.py

# Compare results
python scripts/6_compare_models.py

# Generate visualizations
python scripts/7_visualize_results.py

# Profile models
python scripts/8_profile_model.py
```

### Run Complete Pipeline

```bash
bash run_all.sh
```

---

## 📊 Actual Benchmark Results

The following results are based on the measured benchmark and accuracy evaluation for this project.

### Quantization Results

| Model    | Size        | Mean Latency  | Speedup   | mAP@0.50:0.95 | Accuracy Drop |
| -------- | ----------- | ------------- | --------- | ------------- | ------------- |
| **FP32** | 27.71 MB    | 439.29 ms     | 1.00×     | **37.2%**     | 0.0 pp        |
| **FP16** | 13.92 MB    | 296.62 ms     | **1.48×** | **37.1%**     | 0.1 pp        |
| **INT8** | **7.26 MB** | **249.61 ms** | **1.76×** | **35.4%**     | **1.8 pp**    |

### Model Size Reduction

| Model | Size        | Reduction vs FP32 |
| ----- | ----------- | ----------------- |
| FP32  | 27.71 MB    | 1.00×             |
| FP16  | 13.92 MB    | **1.99×**         |
| INT8  | **7.26 MB** | **3.82×**         |

### Latency Statistics

| Metric    | FP32      | FP16      | INT8          |
| --------- | --------- | --------- | ------------- |
| Mean      | 439.29 ms | 296.62 ms | **249.61 ms** |
| Median    | 447.33 ms | 366.10 ms | **247.95 ms** |
| Std. Dev. | 45.36 ms  | 144.81 ms | **14.30 ms**  |
| P95       | 483.66 ms | 464.34 ms | **267.52 ms** |
| P99       | 494.05 ms | 478.76 ms | **300.16 ms** |

### Key Takeaway

**INT8 is the strongest optimization in this benchmark:**

* **3.82× smaller model**
* **1.76× mean latency speedup**
* **43.2% lower mean inference latency**
* **44.7% lower P95 latency**
* **1.8 percentage-point mAP@0.50:0.95 drop**

FP16 provides an accuracy-focused alternative:

* **1.99× smaller model**
* **1.48× mean latency speedup**
* Only **0.1 percentage-point mAP@0.50:0.95 drop**

> **Note:** The measured results do not demonstrate a 4× inference speedup. The actual INT8 speedup is **1.76×** on the tested CPU configuration.

---

## 📁 Generated Files

After running the pipeline, the following files are generated:

```text
results/

├── benchmark_results.json       # Raw latency metrics
├── accuracy_results.json        # mAP metrics
├── combined_results.csv         # Combined model comparison
├── SUMMARY.md                   # Comprehensive results report
│
└── plots/
    ├── latency_comparison.png
    ├── accuracy_comparison.png
    ├── size_comparison.png
    └── tradeoff_curve.png
```

---

## 🔧 Customization

### Change Model Variant

Edit the model configuration in:

```text
scripts/1_download_model.py
```

Example:

```python
MODEL_NAME = 'yolov5n'  # Nano - smaller/faster
MODEL_NAME = 'yolov5s'  # Small - default
MODEL_NAME = 'yolov5m'  # Medium
MODEL_NAME = 'yolov5l'  # Large
```

### Adjust Benchmark Parameters

Edit:

```text
scripts/4_benchmark_latency.py
```

Example:

```python
NUM_RUNS = 100       # Increase for better statistical confidence
WARMUP_RUNS = 10     # Warm-up before measurement
```

### Adjust Accuracy Evaluation

Edit:

```text
scripts/5_evaluate_accuracy.py
```

Example:

```python
NUM_IMAGES = 1000    # Increase for more extensive evaluation
```

### Change Quantization Settings

Edit:

```text
scripts/3_convert_to_tflite.py
```

Example:

```python
NUM_CALIBRATION_SAMPLES = 200
```

Increasing the number and diversity of calibration images can improve the representativeness of INT8 calibration.

---

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution:**

* Reduce `NUM_IMAGES` during accuracy evaluation
* Use a smaller YOLOv5 variant
* Reduce the number of simultaneous processes
* Close other memory-intensive applications

### Issue: Slow Execution

**Solution:**

* Reduce `NUM_RUNS` during development
* Reduce the number of evaluation images
* Use fewer calibration samples during experimentation
* Use the YOLOv5n model for faster testing

### Issue: COCO Download Fails

**Solution:**

* Check the internet connection
* Retry the setup script
* Manually download the required COCO validation data
* Verify the dataset path configured in the project

### Issue: Model Conversion Fails

**Solution:**

* Check TensorFlow and PyTorch versions
* Reinstall dependencies from `requirements.txt`
* Verify the downloaded YOLOv5 model
* Check that sufficient RAM and disk space are available

---

## 📖 Additional Resources

* **Full Documentation:** See `README.md`
* **Results Analysis:** See `results/SUMMARY.md` after running the pipeline
* **Code Documentation:** Review comments and documentation within the scripts
* **Interview Preparation:** See the optimization and benchmarking sections in `README.md`

---

## 💡 Tips for Interviews

When discussing this project, use the **actual measured results** rather than the earlier illustrative figures.

### 1. Explain the Main Result

> "I optimized a YOLOv5s object detection model using TensorFlow Lite post-training quantization. The INT8 version reduced the model size from 27.71 MB to 7.26 MB, a 3.82× reduction, and reduced mean CPU inference latency from 439.29 ms to 249.61 ms, giving a measured 1.76× speedup."

### 2. Explain the Accuracy Trade-off

> "The INT8 model achieved 35.4% mAP@0.50:0.95 compared with 37.2% for FP32, resulting in a 1.8 percentage-point accuracy drop."

### 3. Discuss FP16

> "FP16 reduced the model size by approximately 2× and achieved a 1.48× mean latency improvement while maintaining almost the same detection accuracy, with only a 0.1 percentage-point mAP drop."

### 4. Discuss the Optimization Techniques

Be prepared to explain:

* Post-training quantization
* FP16 vs INT8 quantization
* Representative dataset calibration
* Scale and zero-point
* CPU integer kernels
* Memory bandwidth
* SIMD/vectorized operations
* Model-size reduction

### 5. Discuss Benchmarking

Explain why you measured more than just average latency:

* Mean latency
* Median latency
* Standard deviation
* P95 latency
* P99 latency
* Warm-up runs

For example:

> "I used percentile-based latency analysis because average latency alone doesn't show tail behavior. For INT8, the median was 247.95 ms, P95 was 267.52 ms and P99 was 300.16 ms."

### 6. Discuss Deployment Trade-offs

A strong interview answer is:

> "FP32 is the baseline when accuracy is the priority. FP16 provides nearly lossless accuracy with roughly 2× compression. INT8 gives the smallest model and best measured CPU latency, but introduces a slightly larger accuracy trade-off. The appropriate choice depends on the target hardware and application requirements."

### 7. Avoid Overclaiming

Do **not** claim:

> "INT8 gives a universal 4× speedup."

Instead say:

> "On my tested CPU configuration, INT8 achieved a measured 1.76× speedup."

Similarly, avoid claiming a universal 75% cost reduction from the benchmark. The project measures **latency and model-size improvements**, not direct infrastructure cost savings.

---

## ⭐ Project Summary

| Category                          | Best Result        |
| --------------------------------- | ------------------ |
| **Smallest Model**                | INT8 — 7.26 MB     |
| **Size Reduction**                | INT8 — 3.82×       |
| **Lowest Mean Latency**           | INT8 — 249.61 ms   |
| **Best Mean Speedup**             | INT8 — 1.76×       |
| **Best Accuracy Preservation**    | FP16 — 0.1 pp drop |
| **Best Overall CPU Optimization** | **INT8**           |

---

**⭐ If this project helps you, please consider starring it on GitHub!**

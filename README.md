# YOLOv5 TensorFlow Lite Optimization

---

## 🎯 Project Overview

This project demonstrates **neural network inference optimization** using TensorFlow Lite post-training quantization on a pre-trained YOLOv5 object detection model. The goal is to reduce model size and CPU inference latency while maintaining detection accuracy, showcasing practical model deployment and edge-optimization techniques.

Three model variants are generated and evaluated:

- **FP32** — baseline floating-point model
- **FP16** — half-precision model
- **INT8** — integer-quantized model using representative-dataset calibration

### Key Results

| Model Type | Size | Mean Inference Time | Speedup vs FP32 | mAP@0.50:0.95 | Size Reduction |
|------------|------|---------------------|-----------------|---------------|----------------|
| **FP32 (Baseline)** | 27.71 MB | 439.29 ms | 1.00× | 37.2% | 1.00× |
| **FP16** | 13.92 MB | 296.62 ms | 1.48× | 37.1% | 1.99× |
| **INT8** | **7.26 MB** | **249.61 ms** | **1.76×** | **35.4%** | **3.82×** |

### 🎯 Achieved Result

**INT8 quantization reduced the model size by 3.82× and reduced mean CPU inference latency by 43.2%, achieving a measured 1.76× speedup with a 1.8 percentage-point drop in mAP@0.50:0.95.**

FP16 provided an alternative accuracy-focused optimization, reducing model size by approximately 2× and mean latency by approximately 32.5%, while reducing mAP@0.50:0.95 by only 0.1 percentage points.

> **Important:** These figures are the actual measured benchmark results for this project. They replace earlier illustrative claims such as 4× inference speedup, 45 ms latency, and 76.8% mAP.

---

## 🚀 Features

- ✅ **Model Conversion**: PyTorch YOLOv5 → TensorFlow SavedModel → TensorFlow Lite
- ✅ **Quantization**: FP32, FP16, and INT8 post-training quantization
- ✅ **Representative Dataset Calibration**: Calibration for INT8 activation quantization
- ✅ **Comprehensive Benchmarking**: Mean, median, standard deviation, P95 and P99 latency
- ✅ **Accuracy Evaluation**: COCO-style mAP evaluation
- ✅ **Model Comparison**: Latency, accuracy and model-size trade-off analysis
- ✅ **Visualization**: Latency, accuracy and size comparison plots
- ✅ **CPU Optimization**: Designed and benchmarked for CPU inference
- ✅ **Profiling**: Layer-wise and model-level performance analysis

---

## 📁 Project Structure

```text
yolo-tflite-optimization/

├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── setup.sh                           # Setup script
│
├── scripts/
│   ├── 1_download_model.py           # Download pre-trained YOLOv5
│   ├── 2_convert_to_tensorflow.py    # Convert PyTorch → TensorFlow
│   ├── 3_convert_to_tflite.py        # Convert TensorFlow → TFLite + quantization
│   ├── 4_benchmark_latency.py        # Measure inference latency
│   ├── 5_evaluate_accuracy.py        # Evaluate detection accuracy
│   ├── 6_compare_models.py           # Compare model variants
│   ├── 7_visualize_results.py        # Generate comparison plots
│   └── 8_profile_model.py            # Profile model performance
│
├── models/                            # Saved models (generated)
│   ├── yolov5s.pt                     # Original PyTorch model
│   ├── yolov5s_saved_model/           # TensorFlow SavedModel
│   ├── yolov5s_fp32.tflite             # FP32 TFLite model
│   ├── yolov5s_fp16.tflite             # FP16 TFLite model
│   └── yolov5s_int8.tflite             # INT8 TFLite model
│
├── data/                              # Dataset files
│   ├── coco_val2017_subset/            # COCO validation subset
│   └── representative_dataset/        # Calibration images
│
├── results/                           # Benchmark results (generated)
│   ├── benchmark_results.json          # Raw benchmark data
│   ├── accuracy_results.json           # Accuracy metrics
│   ├── combined_results.csv            # Combined results
│   └── plots/                          # Visualization plots
│
└── utils/
    ├── __init__.py
    ├── dataset.py                      # Dataset utilities
    ├── metrics.py                      # Evaluation metrics
    └── visualization.py                # Plotting utilities
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip
- 4GB+ RAM
- 2GB+ free disk space

### Setup

```bash
# Clone the repository
git clone https://github.com/aviral2309/Neural-Network-Inference-Optimization-TensorFlow-Lite-Deployment-.git

cd yolo-tflite-optimization

# Create virtual environment (recommended)
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
bash setup.sh
```

---

## 🎬 Quick Start

Run the complete optimization pipeline:

```bash
# 1. Download pre-trained YOLOv5 model
python scripts/1_download_model.py

# 2. Convert PyTorch model to TensorFlow
python scripts/2_convert_to_tensorflow.py

# 3. Convert to TensorFlow Lite with quantization
python scripts/3_convert_to_tflite.py

# 4. Benchmark inference latency
python scripts/4_benchmark_latency.py

# 5. Evaluate accuracy on COCO validation set
python scripts/5_evaluate_accuracy.py

# 6. Compare all model variants
python scripts/6_compare_models.py

# 7. Generate visualization plots
python scripts/7_visualize_results.py

# 8. Profile model
python scripts/8_profile_model.py
```

If available in the repository, the complete workflow can also be executed with:

```bash
bash run_all.sh
```

---

## 📊 Optimization Techniques

### 1. Post-Training Quantization

Post-training quantization converts a trained model to lower-precision representations without requiring model retraining.

#### FP16 — Float16 Quantization

Converts 32-bit floating-point values to 16-bit floating-point values.

**Measured result:**

- Model size: **27.71 MB → 13.92 MB**
- Size reduction: **1.99×**
- Mean latency: **439.29 ms → 296.62 ms**
- Speedup: **1.48×**
- mAP@0.50:0.95: **37.2% → 37.1%**
- Accuracy drop: **0.1 percentage points**

#### INT8 — Integer Quantization

Converts the model to 8-bit integer representations using calibration data.

**Measured result:**

- Model size: **27.71 MB → 7.26 MB**
- Size reduction: **3.82×**
- Mean latency: **439.29 ms → 249.61 ms**
- Speedup: **1.76×**
- mAP@0.50:0.95: **37.2% → 35.4%**
- Accuracy drop: **1.8 percentage points**

**Advantages:**

- Strongest model-size reduction
- Lowest measured mean latency
- Reduced memory footprint
- Suitable for CPU and resource-constrained deployment

**Trade-off:**

- Larger accuracy degradation than FP16

---

### 2. Representative Dataset Calibration

INT8 quantization requires representative data to estimate activation ranges and determine quantization parameters such as scale and zero-point.

The calibration process:

1. Select representative images from the deployment data distribution.
2. Run the images through the model.
3. Collect activation statistics.
4. Determine appropriate quantization parameters.
5. Convert the model to the quantized representation.

Conceptually:

```text
real_value = (quantized_value - zero_point) × scale

quantized_value =
    round(real_value / scale) + zero_point
```

---

### 3. Benchmarking Methodology

The project evaluates inference performance using repeated measurements rather than a single timing sample.

### Reported Metrics

- **Mean latency**
- **Median latency**
- **Standard deviation**
- **P95 latency**
- **P99 latency**

### Actual Results

| Metric | FP32 | FP16 | INT8 |
|--------|------|------|------|
| Mean | 439.29 ms | 296.62 ms | **249.61 ms** |
| Median | 447.33 ms | 366.10 ms | **247.95 ms** |
| Std. Dev. | 45.36 ms | 144.81 ms | **14.30 ms** |
| P95 | 483.66 ms | 464.34 ms | **267.52 ms** |
| P99 | 494.05 ms | 478.76 ms | **300.16 ms** |

Warm-up inference is used before collecting measurements to reduce startup effects and improve benchmark consistency.

> **Observation:** FP16 has substantially higher latency variability in this benchmark, as reflected by its 144.81 ms standard deviation. Therefore, mean latency should be interpreted together with median and percentile metrics.

---

## 🎯 Accuracy Evaluation

The project evaluates object-detection accuracy using mAP metrics.

### Actual Accuracy Results

| Metric | FP32 | FP16 | INT8 |
|--------|------|------|------|
| **mAP@0.50** | 56.0% | 55.9% | 54.1% |
| **mAP@0.75** | 39.8% | 39.7% | 38.0% |
| **mAP@0.50:0.95** | **37.2%** | **37.1%** | **35.4%** |
| **Drop vs FP32** | 0.0 pp | **0.1 pp** | **1.8 pp** |

### Interpretation

**FP16** preserves almost all of the baseline detection accuracy.

**INT8** introduces a larger accuracy trade-off, but the model remains substantially smaller and faster than FP32 on the tested CPU configuration.

---

## 📈 Results Interpretation

### Model Size

```text
FP32  ████████████████████████████  27.71 MB
FP16  ██████████████                13.92 MB
INT8  ███████                        7.26 MB
```

INT8 reduces the model size by approximately **73.8%** compared with FP32.

FP16 reduces the model size by approximately **49.8%**.

### Mean Latency

```text
FP32  ████████████████████████████████████████  439.29 ms
FP16  ███████████████████████████               296.62 ms
INT8  ███████████████████████                   249.61 ms
```

Compared with FP32:

- FP16 reduces mean latency by approximately **32.5%**
- INT8 reduces mean latency by approximately **43.2%**

### Latency vs Accuracy Trade-off

```text
mAP@0.50:0.95

37.2% ● FP32
      │
37.1% ● FP16
      │
      │
35.4% ● INT8
      └────────────────────────────
       439 ms    297 ms    250 ms
              Mean Latency
```

The results demonstrate a practical optimization trade-off:

- **FP32** → baseline accuracy and largest model
- **FP16** → approximately 2× smaller with almost unchanged accuracy
- **INT8** → smallest model and lowest measured latency, with a modest accuracy trade-off

---

## ⚡ Why INT8 Improves CPU Efficiency

INT8 quantization can improve CPU inference performance through:

### 1. Reduced Memory Traffic

INT8 values use fewer bits than FP32 values, reducing the amount of data transferred through the memory hierarchy.

### 2. Optimized Integer Kernels

CPU inference runtimes can use optimized integer kernels and SIMD/vector instructions for quantized operations.

### 3. Improved Cache Efficiency

The smaller representation can reduce memory pressure and improve cache utilization.

### 4. Efficient Integer Computation

Modern CPUs can execute optimized integer operations efficiently, depending on the processor architecture and runtime.

> The exact speedup depends on the CPU, inference runtime, delegate, thread configuration and model architecture. Therefore, this project reports the measured **1.76× INT8 speedup** rather than claiming a universal 4× improvement.

---

## 🔬 Technical Deep Dive

### Quantization Process

A representative INT8 conversion can be configured using TensorFlow Lite's converter:

```python
def representative_dataset_gen():
    for image in calibration_images:
        yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model(
    saved_model_dir
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

tflite_model = converter.convert()
```

The exact input/output data types should match the conversion and inference pipeline implemented in the project.

---

## 🏆 Project Highlights

### Performance Optimization

- **3.82× model-size reduction** with INT8
- **1.76× measured mean CPU speedup** with INT8
- **43.2% reduction in mean inference latency**
- **44.7% reduction in P95 latency** with INT8

### Accuracy Preservation

- FP16: **0.1 pp mAP@0.50:0.95 drop**
- INT8: **1.8 pp mAP@0.50:0.95 drop**

### Benchmarking

- Mean latency
- Median latency
- Standard deviation
- P95 latency
- P99 latency

### Deployment Skills

- PyTorch → TensorFlow → TFLite conversion
- Post-training quantization
- Representative dataset calibration
- CPU inference optimization
- Object detection evaluation
- Performance/accuracy trade-off analysis

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **Model Conversion** — Cross-framework conversion from PyTorch to TensorFlow Lite.
2. **Quantization Techniques** — Understanding FP16 and INT8 precision trade-offs.
3. **Benchmarking Best Practices** — Using repeated measurements and latency percentiles.
4. **Performance Optimization** — Reducing model footprint and CPU inference latency.
5. **Accuracy Evaluation** — Measuring quantization impact using object-detection metrics.
6. **Deployment Optimization** — Selecting model variants based on hardware and application requirements.
7. **Edge AI Concepts** — Understanding the trade-offs involved in deploying neural networks on resource-constrained hardware.

---

## 🧭 When to Use Each Variant

| Variant | Best Use Case | Main Advantage | Main Trade-off |
|---------|---------------|----------------|----------------|
| **FP32** | Accuracy-first applications | Baseline accuracy | Largest model and highest latency |
| **FP16** | Accuracy + compression | ~2× smaller with almost no mAP loss | Latency improvement is hardware-dependent |
| **INT8** | CPU / edge deployment | Smallest model and lowest measured latency | 1.8 pp mAP@0.50:0.95 drop |

---

## 📚 References

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- [COCO Dataset](https://cocodataset.org/)
- [Post-Training Quantization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- Ultralytics for YOLOv5
- TensorFlow team for TensorFlow Lite
- COCO dataset creators

---

## 📧 Contact

**Aviral Mittal**

- Email: aviralmittal0012@gmail.com
- GitHub: [@aviral2309](https://github.com/aviral2309)

---

**This project demonstrates practical deep-learning inference optimization through model conversion, quantization, benchmarking, accuracy evaluation and deployment-oriented performance analysis.**

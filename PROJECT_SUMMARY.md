# Project Summary: YOLOv5 TensorFlow Lite Optimization

---

## 🎯 Project Goal

Demonstrate neural network inference optimization using TensorFlow Lite post-training quantization on a YOLOv5s object detection model, with a focus on reducing model size and CPU inference latency while preserving detection accuracy for edge and resource-constrained deployment.

The project evaluates three deployment variants:

- **FP32** — baseline floating-point model
- **FP16** — half-precision model
- **INT8** — fully integer-quantized model with representative-dataset calibration

---

## 📊 Actual Benchmark Results

| Metric | FP32 (Baseline) | FP16 | INT8 |
|--------|------------------|------|------|
| **Model Size** | 27.71 MB | 13.92 MB | **7.26 MB** |
| **Size Reduction** | 1.00× | **1.99×** | **3.82×** |
| **Mean Latency** | 439.29 ms | 296.62 ms | **249.61 ms** |
| **Speedup vs FP32** | 1.00× | **1.48×** | **1.76×** |
| **Median Latency** | 447.33 ms | 366.10 ms | **247.95 ms** |
| **P95 Latency** | 483.66 ms | 464.34 ms | **267.52 ms** |
| **P99 Latency** | 494.05 ms | 478.76 ms | **300.16 ms** |
| **mAP@0.50** | 56.0% | 55.9% | **54.1%** |
| **mAP@0.75** | 39.8% | 39.7% | **38.0%** |
| **mAP@0.50:0.95** | 37.2% | 37.1% | **35.4%** |
| **mAP Drop vs FP32** | 0.0 pp | **0.1 pp** | **1.8 pp** |

### Key Gains

### INT8

- **3.82× smaller model**: 27.71 MB → 7.26 MB
- **1.76× lower mean inference latency**: 439.29 ms → 249.61 ms
- **1.80 percentage-point mAP@0.50:0.95 drop**: 37.2% → 35.4%
- **P95 latency reduced by ~44.7%**: 483.66 ms → 267.52 ms
- **P99 latency reduced by ~39.2%**: 494.05 ms → 300.16 ms

### FP16

- **1.99× smaller model**: 27.71 MB → 13.92 MB
- **1.48× lower mean latency**: 439.29 ms → 296.62 ms
- **Only 0.1 percentage-point mAP@0.50:0.95 drop**: 37.2% → 37.1%
- The FP16 latency results show **high variability** (144.81 ms standard deviation), so the mean improvement should be interpreted together with median and percentile measurements.

> **Main Technical Insight:** INT8 provides the best overall compression and the strongest CPU latency improvement among the tested variants, while maintaining most of the baseline detection accuracy. FP16 provides almost no measurable accuracy loss and nearly halves the model size, but its latency is more variable in this benchmark.

---

## 🏗️ Technical Architecture

### Pipeline Overview

```text
PyTorch YOLOv5
       ↓
TensorFlow SavedModel
       ↓
TFLite Conversion
       ↓
┌──────────────┬──────────────┬──────────────┐
│     FP32     │     FP16     │     INT8     │
│   Baseline   │ Half Float   │ Quantized   │
└──────────────┴──────────────┴──────────────┘
       ↓
Latency Benchmarking
       ↓
COCO Accuracy Evaluation
       ↓
Model Comparison
       ↓
Visualization & Profiling
```

### Core Technologies

- **Model:** YOLOv5s object detection
- **Framework:** TensorFlow Lite / LiteRT
- **Source Framework:** PyTorch
- **Optimization:** Post-Training Quantization (PTQ)
- **Quantization:** FP16 and INT8
- **Calibration:** Representative dataset
- **Evaluation:** COCO mAP metrics
- **Benchmarking:** Mean, median, standard deviation, P95 and P99 latency
- **Deployment Focus:** CPU / edge inference

---

## 📁 Code Structure

### Scripts (8 Total)

1. **`1_download_model.py`**
   - Downloads pre-trained YOLOv5 weights from Ultralytics
   - Verifies model integrity
   - Supports different YOLOv5 variants

2. **`2_convert_to_tensorflow.py`**
   - Converts PyTorch YOLOv5 to TensorFlow SavedModel
   - Supports YOLOv5 export / ONNX-based conversion
   - Validates the converted model

3. **`3_convert_to_tflite.py`**
   - **Core optimization script**
   - Generates FP32, FP16 and INT8 TFLite variants
   - Implements representative-dataset calibration for INT8

4. **`4_benchmark_latency.py`**
   - Performs repeated inference benchmarking
   - Uses warm-up runs
   - Calculates mean, median, standard deviation, P95 and P99 latency

5. **`5_evaluate_accuracy.py`**
   - Evaluates detection accuracy
   - Calculates COCO-style mAP
   - Performs NMS post-processing
   - Quantifies accuracy degradation after quantization

6. **`6_compare_models.py`**
   - Combines latency, size and accuracy metrics
   - Calculates speedup and model-size reduction
   - Helps identify deployment trade-offs

7. **`7_visualize_results.py`**
   - Generates comparison plots
   - Visualizes latency, model size and accuracy trade-offs
   - Produces Markdown-based result summaries

8. **`8_profile_model.py`**
   - Performs layer-wise profiling
   - Analyzes memory usage
   - Identifies computational bottlenecks

### Utility Modules

- **`utils/dataset.py`** — dataset handling, preprocessing and calibration data generation
- **`utils/metrics.py`** — IoU, NMS, mAP and statistical calculations
- **`utils/visualization.py`** — latency, size, accuracy and trade-off visualization

---

## 🔬 Key Optimization Techniques

### 1. Post-Training Quantization (PTQ)

The project applies quantization after training, avoiding the need to retrain the YOLOv5 model.

#### FP16 Quantization

Weights are converted from 32-bit floating point to 16-bit floating point:

```python
converter.target_spec.supported_types = [tf.float16]
```

This reduced the model size from **27.71 MB to 13.92 MB**, approximately **1.99× smaller**.

### 2. INT8 Quantization

INT8 quantization converts model computations/weights to 8-bit integer representations and uses a representative dataset to calibrate activation ranges.

```python
converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
```

The result was:

- **27.71 MB → 7.26 MB**
- **3.82× model-size reduction**
- **439.29 ms → 249.61 ms mean latency**
- **1.76× mean speedup**
- **1.8 percentage-point mAP@0.50:0.95 reduction**

### 3. Representative Dataset Calibration

Calibration is used to estimate appropriate quantization parameters such as scale and zero-point for activations.

Conceptually:

```text
real_value = (quantized_value - zero_point) × scale

quantized_value =
    round(real_value / scale) + zero_point
```

A representative calibration dataset should reflect the distribution of data expected during deployment.

---

## 📈 Detailed Performance Analysis

### Model Size

```text
FP32  ████████████████████████████  27.71 MB
FP16  ██████████████                13.92 MB
INT8  ███████                        7.26 MB
```

INT8 achieves the strongest compression, reducing storage requirements by approximately **73.8%** compared with FP32.

FP16 reduces storage by approximately **49.8%**.

### Mean Inference Latency

```text
FP32  ████████████████████████████████████████  439.29 ms
FP16  ███████████████████████████               296.62 ms
INT8  ███████████████████████                   249.61 ms
```

Compared with FP32:

- FP16 improves mean latency by approximately **32.5%**
- INT8 improves mean latency by approximately **43.2%**

### P95 Latency

| Model | P95 Latency |
|-------|-------------|
| FP32 | 483.66 ms |
| FP16 | 464.34 ms |
| **INT8** | **267.52 ms** |

INT8 provides a substantial reduction in tail latency, which is important for consistent real-time or interactive inference workloads.

---

## 🎯 Accuracy vs Performance Trade-off

### mAP@0.50:0.95

| Model | mAP@0.50:0.95 | Drop |
|-------|---------------|------|
| **FP32** | **37.2%** | Baseline |
| **FP16** | **37.1%** | 0.1 pp |
| **INT8** | **35.4%** | 1.8 pp |

FP16 preserves almost all of the baseline accuracy.

INT8 introduces a larger but still relatively small degradation of **1.8 percentage points** in mAP@0.50:0.95.

### Trade-off Summary

```text
Accuracy
  │
37.2% ●────────────── FP32
  │    \
37.1% │  ●─────────── FP16
  │
35.4% │       ●────── INT8
  │
  └──────────────────────────
       Latency
      439    297    250 ms
```

The results demonstrate the classic deployment trade-off:

**FP32 → maximum baseline accuracy**

**FP16 → strong size reduction with almost unchanged accuracy**

**INT8 → smallest model and best measured CPU latency, with a modest accuracy trade-off**

---

## ⚡ Important Benchmark Observation

The results do **not** support claiming a 4× inference speedup.

The measured mean latency is:

- FP32: **439.29 ms**
- FP16: **296.62 ms**
- INT8: **249.61 ms**

Therefore, the measured INT8 speedup is **1.76×**, not 4×.

Similarly, the model-size reduction is **3.82×**, not exactly 4×.

This distinction is important when presenting the project in interviews or on a resume because the results should match the actual benchmark output.

---

## 🧠 Why INT8 Performs Better

INT8 can improve CPU inference efficiency through several mechanisms:

### 1. Reduced Memory Traffic

INT8 values require substantially less storage than FP32 values, reducing the amount of data that must be moved through the memory hierarchy.

### 2. Efficient Integer Kernels

Modern CPU inference libraries can use optimized integer kernels and SIMD/vector instructions for quantized operations.

### 3. Better Cache Utilization

The smaller quantized representation can improve cache efficiency and reduce memory pressure.

### 4. Lower Computational Cost

Integer arithmetic can be highly efficient on CPUs that provide optimized INT8 execution paths.

> The exact speedup depends strongly on the CPU, TensorFlow Lite runtime, delegate, number of threads, model architecture and benchmark configuration. Therefore, the measured **1.76× speedup** should be presented as the result obtained on the tested hardware/configuration rather than as a universal INT8 speedup.

---

## 📊 Benchmark Methodology

The benchmark records multiple latency statistics instead of relying on a single inference measurement:

- **Mean**
- **Median**
- **Standard deviation**
- **P95**
- **P99**

Warm-up inference is used before collecting benchmark measurements to reduce startup effects such as cache initialization and runtime setup.

### Why Percentiles Matter

Mean latency alone can hide occasional slow inferences.

For example, INT8 achieved:

- Median: **247.95 ms**
- P95: **267.52 ms**
- P99: **300.16 ms**

This gives a better picture of both typical and tail inference performance.

---

## 🚀 Deployment Recommendations

### INT8 — Best for CPU / Edge Optimization

Recommended when the priority is:

- Small model footprint
- Lower CPU latency
- Lower memory bandwidth requirements
- Edge and resource-constrained deployment
- Better throughput than FP32

The measured result provides **3.82× compression** and **1.76× mean latency improvement** with a **1.8 pp mAP drop**.

### FP16 — Best Accuracy/Compression Balance

Recommended when:

- Accuracy preservation is highly important
- Hardware/runtime supports efficient FP16 execution
- Approximately 2× model-size reduction is useful
- Some latency improvement is desired

In this benchmark, FP16 retains almost the same mAP as FP32, with only a **0.1 pp drop**.

### FP32 — Baseline

Recommended when:

- Maximum baseline numerical precision is required
- Model size and latency are less important
- The deployment hardware is optimized for FP32

---

## 🎓 Interview Talking Points

### 1. Quantization

> "I took a pre-trained YOLOv5s object detection model and converted it into TensorFlow Lite FP32, FP16 and INT8 variants using post-training quantization."

### 2. INT8 Optimization

> "Using representative-dataset calibration, I converted the model to INT8 and reduced its size from 27.71 MB to 7.26 MB, which is a 3.82× reduction."

### 3. Performance Gain

> "On the tested CPU configuration, INT8 reduced mean inference latency from 439.29 ms to 249.61 ms, giving a measured 1.76× speedup."

### 4. Accuracy Trade-off

> "The INT8 model achieved 35.4% mAP@0.50:0.95 compared with 37.2% for FP32, corresponding to a 1.8 percentage-point drop."

### 5. FP16 Comparison

> "FP16 preserved accuracy extremely well, with only a 0.1 percentage-point mAP drop, while reducing the model size by approximately 2×."

### 6. Benchmarking

> "Instead of reporting one inference time, I measured mean, median, standard deviation, P95 and P99 latency to characterize both normal and tail performance."

---

## 💼 Resume-Ready Project Description

**YOLOv5 TensorFlow Lite Optimization | Python, TensorFlow Lite, PyTorch, Computer Vision**

- Optimized a YOLOv5s object detection model using **FP16 and INT8 post-training quantization**, creating lightweight TFLite variants for CPU/edge deployment.
- Reduced model size from **27.71 MB to 7.26 MB (3.82×)** with INT8 quantization while maintaining **35.4% mAP@0.50:0.95** vs **37.2% FP32 baseline**.
- Reduced mean CPU inference latency from **439.29 ms to 249.61 ms**, achieving a measured **1.76× speedup** with INT8.
- Built an evaluation pipeline covering **latency statistics, COCO mAP, NMS, model-size analysis, profiling and accuracy-performance trade-offs**.

---

## 📌 Project Takeaway

This project demonstrates the practical optimization of a deep learning object detection model for resource-constrained inference.

The most important measured result is:

> **INT8 reduced model size by 3.82× and mean CPU inference latency by 1.76×, while reducing mAP@0.50:0.95 by only 1.8 percentage points.**

The project therefore demonstrates practical skills in:

- Deep learning model deployment
- TensorFlow Lite conversion
- Post-training quantization
- INT8 calibration
- CPU inference optimization
- Computer vision
- Performance benchmarking
- Accuracy-performance trade-off analysis
- Edge AI deployment

---

## 📧 Contact

**Aviral**  
*Electrical Engineering | SGSITS*  
*Focused on Machine Learning, System Design, and AI/ML Infrastructure*

---

**Note:** All performance and accuracy figures in this report are based on the benchmark results supplied for this project. The measured results should be used instead of the earlier illustrative figures such as 4× speedup, 45 ms latency or 76.8% mAP.

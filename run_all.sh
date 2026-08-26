#!/bin/bash
echo "======================================================================"
echo "YOLOv5 TFLite Optimization - Complete Pipeline"
echo "======================================================================"
echo "======================================================================"
echo ""

# Function to run a script and check for errors
run_script() {
    script_name=$1
    script_description=$2
    
    echo ""
    echo "======================================================================"
    echo "Step $3: $script_description"
    echo "======================================================================"
    echo "Running: $script_name"
    echo ""
    
    python3 "$script_name"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ Error: $script_name failed"
        echo "Please check the error messages above and try again."
        exit 1
    fi
    
    echo ""
    echo "✓ Step $3 completed successfully"
    echo ""
    sleep 2
}

# Start time
start_time=$(date +%s)

# Run all scripts in sequence
run_script "scripts/1_download_model.py" "Download Pre-trained YOLOv5 Model" "1/8"
run_script "scripts/2_convert_to_tensorflow.py" "Convert PyTorch to TensorFlow" "2/8"
run_script "scripts/3_convert_to_tflite.py" "Convert to TFLite with Quantization" "3/8"
run_script "scripts/4_benchmark_latency.py" "Benchmark Inference Latency" "4/8"
run_script "scripts/5_evaluate_accuracy.py" "Evaluate Model Accuracy" "5/8"
run_script "scripts/6_compare_models.py" "Compare All Model Variants" "6/8"
run_script "scripts/7_visualize_results.py" "Generate Visualizations" "7/8"
run_script "scripts/8_profile_model.py" "Profile Models" "8/8"

# End time
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "======================================================================"
echo "🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY! 🎉"
echo "======================================================================"
echo ""
echo "Total execution time: ${minutes}m ${seconds}s"
echo ""
echo "Generated artifacts:"
echo "  📁 models/"
echo "     - yolov5s.pt (PyTorch model)"
echo "     - yolov5s_saved_model/ (TensorFlow SavedModel)"
echo "     - yolov5s_fp32.tflite (Baseline)"
echo "     - yolov5s_fp16.tflite (2x faster)"
echo "     - yolov5s_int8.tflite (4x faster)"
echo ""
echo "  📊 results/"
echo "     - benchmark_results.json/csv (Latency metrics)"
echo "     - accuracy_results.json (mAP metrics)"
echo "     - combined_results.json/csv (All metrics)"
echo "     - profiling_results.json (Profiling data)"
echo "     - SUMMARY.md (Comprehensive report)"
echo ""
echo "  📈 results/plots/"
echo "     - latency_comparison.png"
echo "     - accuracy_comparison.png"
echo "     - size_comparison.png"
echo "     - tradeoff_curve.png"
echo "     - comparison_table.csv"
echo ""
echo "Key Results:"
if [ -f "results/combined_results.json" ]; then
    echo "  View comprehensive report: results/SUMMARY.md"
    echo "  View plots: results/plots/*.png"
fi
echo ""
echo "======================================================================"
echo "Next steps:"
echo "  • Review SUMMARY.md for complete analysis"
echo "  • View plots in results/plots/ directory"
echo "  • Examine JSON files for detailed metrics"
echo "  • Upload to GitHub: git add . && git commit -m 'Complete optimization' && git push"
echo "======================================================================"

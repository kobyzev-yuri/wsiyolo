# WSI YOLO Pipeline

A comprehensive pipeline for processing Whole Slide Images (WSI) using YOLO object detection models. This project implements a complete workflow for analyzing histopathology images with multiple YOLO models and merging overlapping predictions.

## 🎯 Project Overview

This pipeline is designed for histopathology analysis, specifically for detecting and classifying different types of tissue regions in large medical images. The system uses multiple YOLO models to detect various pathological features and intelligently merges overlapping predictions to provide accurate results.

### Key Features

- **Multi-Model YOLO Detection**: Support for multiple YOLO models with different confidence thresholds
- **Intelligent Patch Extraction**: Automatic extraction of tissue patches from WSI files
- **Complete Patch Coverage**: Extracts ALL patches containing predictions, not just a limited subset
- **Background Filtering**: Advanced background detection to skip non-tissue regions
- **Prediction Merging**: Smart merging of overlapping predictions using IoU thresholds
- **Comprehensive Statistics**: Detailed analysis and reporting of detection results
- **Batch Processing**: Support for processing multiple WSI files

## 🏗️ Architecture & Core Ideas

### 1. **Modular Pipeline Design**
The system is built with a modular architecture that separates concerns:

- **WSI Pipeline**: Handles WSI loading and patch extraction using MONAI
- **YOLO Inference**: Manages YOLO model predictions with configurable parameters
- **Polygon Merger**: Intelligently merges overlapping predictions
- **Data Structures**: Clean, typed data structures for all components

### 2. **Smart Patch Extraction**
The pipeline uses sophisticated patch extraction strategies:

- **Overlapping Windows**: Configurable overlap ratio to ensure no features are missed
- **Background Detection**: Multiple algorithms to identify and skip background regions:
  - HSV-based tissue detection
  - Saturation threshold analysis
  - Morphological operations for noise reduction
- **Memory Optimization**: Efficient handling of large WSI files

### 3. **Multi-Model Ensemble**
The system supports multiple YOLO models with different configurations:

- **Model-Specific Parameters**: Each model has its own confidence threshold and window size
- **Parallel Processing**: Models can be run independently on the same patches
- **Flexible Configuration**: Easy addition of new models through configuration

### 4. **Intelligent Prediction Merging**
Advanced algorithms for handling overlapping predictions:

- **IoU-Based Merging**: Uses Intersection over Union to identify overlapping detections
- **Polygon Union**: Combines overlapping polygons using geometric operations
- **Confidence Weighting**: Considers confidence scores in merging decisions

## 📁 Project Structure

```
wsiyolo/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── data_structures.py        # Data classes and structures
│   ├── main.py                   # Main pipeline implementation
│   ├── monai_pipeline.py         # WSI processing with MONAI
│   ├── polygon_merger.py         # Prediction merging logic
│   ├── simple_patch_loader.py    # Basic patch loading
│   ├── wsi_patch_loader.py       # Advanced WSI patch extraction
│   └── yolo_inference.py         # YOLO model inference
├── models/                       # YOLO model files (.pt)
├── wsi/                         # WSI image files
├── results/                     # Output results
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Main execution script
├── process_all_wsi.py           # Batch processing script
└── README.md                    # This documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Sufficient RAM for large WSI files (8GB+ recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd wsiyolo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your models**:
   - Place your YOLO model files (.pt) in the `models/` directory
   - The system will automatically detect and configure models

4. **Prepare WSI files**:
   - Place your WSI files (.tiff, .tif, .svs, .ndpi) in the `wsi/` directory

### Basic Usage

#### Single WSI Processing

```bash
python run_pipeline.py
```

This will:
- Process the WSI file in `wsi/19_ibd_mod_S037__20240822_091343.tiff`
- Use all models in the `models/` directory
- Save results to `results/predictions.json`

#### Batch Processing

```bash
python process_all_wsi.py
```

This will:
- Process all WSI files in the `wsi/` directory
- Generate individual result files for each WSI
- Create a summary report

#### Annotated Patch Creation

After running the pipeline, you can create annotated patches with all predictions:

```bash
# Create annotated patches for all predictions
python create_annotated_patches.py
```

This will:
- Extract ALL patches that contain predictions (not just 10-20)
- Create annotated versions with prediction visualizations
- Use grid-based naming: `wsi_name.patch_i.patch_j_annotated.png`
- Generate comprehensive visualizations and statistics

### Patch Naming Convention

Patches are named using the following format:
- **Base WSI name**: Extracted from the WSI filename
- **Grid coordinates**: `patch_i.patch_j` where i,j are the patch numbers in the 512x512 grid
- **Type suffix**: `_annotated` for annotated versions with predictions

Example: `19_ibd_mod_S037__20240822_091343.patch_23.45_annotated.png`
- WSI: `19_ibd_mod_S037__20240822_091343`
- Grid position: patch (23, 45) in the 512x512 grid
- Type: annotated patch with prediction visualizations

## 🔧 Configuration

### Model Configuration

Models are automatically configured based on their filenames:

- **lp.pt**: Low-power model (confidence: 0.5)
- **mild.pt**: Mild detection model (confidence: 0.6)
- **moderate.pt**: Moderate detection model (confidence: 0.7)

### Pipeline Parameters

Key parameters can be adjusted in the pipeline initialization:

```python
pipeline = WSIYOLOPipeline(
    models_config=models_config,
    tile_size=512,           # Patch size
    overlap_ratio=0.5,       # Overlap between patches
    iou_threshold=0.5         # IoU threshold for merging
)
```

## 📊 Output Format

### Prediction Results

Results are saved in JSON format with the following structure:

```json
{
  "wsi_info": {
    "path": "path/to/wsi.tiff",
    "width": 2048,
    "height": 2048,
    "levels": 4,
    "mpp": 0.25
  },
  "predictions": [
    {
      "class_name": "tissue_type",
      "confidence": 0.85,
      "box": {
        "start": {"x": 100, "y": 100},
        "end": {"x": 200, "y": 200}
      },
      "polygon": [
        {"x": 100, "y": 100},
        {"x": 200, "y": 100},
        {"x": 200, "y": 200},
        {"x": 100, "y": 200}
      ]
    }
  ]
}
```

### Statistics

The pipeline provides comprehensive statistics:

- Total number of predictions
- Predictions per class
- Average confidence scores
- Processing time and performance metrics

## 🧠 Technical Details

### Background Detection Algorithms

The pipeline implements multiple background detection strategies:

1. **HSV-based Detection**:
   ```python
   def _is_background_hsv(img, saturation_threshold=30, tissue_ratio=0.1):
       # Converts to HSV and analyzes saturation
       # Uses OTSU thresholding for automatic threshold selection
       # Applies morphological operations for noise reduction
   ```

2. **Grayscale-based Detection**:
   ```python
   def _is_background(img, threshold_value=200, background_ratio=0.99):
       # Simple threshold-based background detection
       # Fast but less accurate for complex images
   ```

### Prediction Merging

The merging algorithm uses geometric operations:

1. **IoU Calculation**: Computes intersection over union for all prediction pairs
2. **Polygon Union**: Uses Shapely library for geometric operations
3. **Confidence Weighting**: Considers confidence scores in final decisions

### Memory Management

- **Lazy Loading**: WSI files are loaded on-demand
- **Patch Streaming**: Large WSI files are processed in chunks
- **Garbage Collection**: Automatic cleanup of processed patches

## 🔍 Advanced Usage

### Custom Model Configuration

You can create custom model configurations:

```python
models_config = [
    {
        'model_path': 'path/to/model.pt',
        'window_size': 512,
        'min_conf': 0.6
    }
]
```

### Debugging and Visualization

The pipeline includes debugging tools:

- **Patch Visualization**: View extracted patches
- **Prediction Overlay**: Visualize predictions on WSI
- **Statistics Analysis**: Detailed performance metrics

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `tile_size` or use smaller WSI files
2. **Model Loading Errors**: Ensure model files are compatible with your YOLO version
3. **WSI Loading Errors**: Check WSI file format and ensure MONAI can read it

### Performance Optimization

- Use GPU acceleration when available
- Adjust `overlap_ratio` based on your use case
- Consider using smaller `tile_size` for memory-constrained systems

## 📈 Performance Metrics

Typical performance on a modern GPU:
- **Processing Speed**: ~100-500 patches/second
- **Memory Usage**: 4-8GB for typical WSI files
- **Accuracy**: Depends on model quality and WSI characteristics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MONAI for WSI processing capabilities
- Ultralytics for YOLO model support
- Shapely for geometric operations
- OpenCV for image processing utilities

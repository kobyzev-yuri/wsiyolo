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
├── tests/                        # Test and debug scripts
│   ├── README.md                 # Tests documentation
│   ├── test_pipeline.py          # Main pipeline tests
│   ├── test_polygon_fix.py       # Polygon processing tests
│   ├── test_real_data_fix.py     # Real data validation tests
│   ├── debug_pipeline.py         # Pipeline debugging
│   └── analyze_polygon_detailed.py # Polygon analysis
├── visualization/                # Visualization and statistics scripts
│   ├── README.md                 # Visualization documentation
│   ├── create_simple_annotations.py # WSI overview with predictions
│   └── view_statistics.py        # Statistics analysis
├── models/                       # YOLO model files (.pt)
├── wsi/                         # WSI image files
├── results/                     # Output results
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Main execution script
├── process_all_wsi.py           # Batch processing script
├── create_annotated_patches.py  # Annotated patch generation
├── extract_patches_with_predictions.py # All patches extraction
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
- Use grid-based naming: `wsi_name_i_j.png`
- Generate comprehensive visualizations and statistics

#### WSI Overview and Statistics

Create overview visualizations and statistical analysis:

```bash
# Create WSI overview with all predictions
python visualization/create_simple_annotations.py

# View detailed statistics
python visualization/view_statistics.py
```

This will:
- Generate WSI overview images with all predictions overlaid
- Create statistical charts showing class distributions
- Provide detailed analysis of prediction confidence
- Export comprehensive statistics for further analysis

### Patch Naming Convention

Patches are named using the following format:
- **Base WSI name**: Extracted from the WSI filename
- **Grid coordinates**: `_i_j` where i,j are integer patch numbers in the 512x512 grid
- **File extension**: `.png` for annotated patches with predictions

Example: `19_ibd_mod_S037__20240822_091343_23_45.png`
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

### Polygon Simplification Algorithm

The pipeline includes a sophisticated polygon simplification algorithm that reduces complex segmentation masks to a maximum of 60 points while preserving important geometric features.

#### Algorithm Parameters

```python
# Core simplification parameters
max_points = 60                    # Maximum number of points in simplified polygon
min_tolerance = 0.1               # Minimum tolerance for Douglas-Peucker algorithm
max_tolerance = 10.0              # Maximum tolerance for Douglas-Peucker algorithm
max_iterations = 10               # Maximum iterations for binary search
target_ratio = 0.8                # Target point ratio (80% of max_points)
```

#### Algorithm Steps

1. **Initial Check**: If polygon already has ≤60 points, return unchanged
2. **Binary Search**: Find optimal tolerance using binary search (up to 10 iterations)
3. **Validation**: Check if simplified polygon is valid and has >3 points
4. **Fallback Sampling**: If still too many points, use uniform sampling
5. **Quality Check**: Ensure final polygon is valid and maintains topology

#### Known Issues and Limitations

**⚠️ Current Problems:**
- **Topology Loss**: Uniform sampling can break polygon topology
- **Shape Distortion**: Aggressive simplification may alter object shape
- **Invalid Polygons**: Over-simplification can create invalid geometries
- **Suboptimal Tolerance**: Binary search may not find the best tolerance value

**🔧 Debugging Information:**
The algorithm provides detailed logging for troubleshooting:
```python
print(f"   Умное упрощение полигона: {len(coords)} -> {len(polygon)} точек")
print(f"   После умного упрощения: {len(polygon.exterior.coords)} точек")
```

#### Recommended Improvements

1. **Adaptive Tolerance**: Use polygon area to determine optimal tolerance
2. **Key Point Preservation**: Prioritize corners and high-curvature points
3. **Quality Metrics**: Compare area before/after simplification
4. **Alternative Methods**: Implement different simplification strategies as fallbacks

#### Testing and Validation

To test the polygon simplification algorithm:

```bash
# Run polygon-specific tests
python tests/test_polygon_fix.py

# Analyze polygon processing in detail
python tests/analyze_polygon_detailed.py
```

## 📊 Algorithm Flowchart

For a detailed visual representation of the current multi-model YOLO prediction algorithm, see:

- **[Complete Algorithm Flowchart](docs/algorithm_flowchart.md)** - Comprehensive diagram of all pipeline stages
- **[Polygon Simplification Details](docs/polygon_simplification.md)** - Detailed polygon processing algorithm
- **[Model Application Strategies](docs/current_algorithm_flowchart.md)** - Current vs optimized model processing approaches

The flowchart documents:
- **Current sequential processing** (patch-by-patch, model-by-model)
- **Performance bottlenecks** and optimization opportunities  
- **Recommended batch processing strategies**
- **Expected performance improvements** (20-50x speedup)

## 🤖 Model Application Strategy

### Current Algorithm (Sequential Processing)

The current pipeline applies models **sequentially for each patch**:

```python
# Current approach in main.py:89-95
for patch in tqdm(patches, desc="Обработка патчей"):
    try:
        predictions = self.yolo_inference.predict_patch(patch)  # All models for one patch
        all_predictions.extend(predictions)
    except Exception as e:
        print(f"⚠️  Ошибка обработки патча {patch.patch_id}: {e}")
        continue
```

**Algorithm Flow:**
1. **For each patch** (N patches total)
2. **Apply all models** to the patch (M models)
3. **Collect predictions** from all models
4. **Move to next patch**

**Time Complexity:** O(N × M) where N = patches, M = models

### Model Processing Details

```python
# In yolo_inference.py:72-109
def predict_patch(self, patch_info: PatchInfo) -> List[Prediction]:
    all_predictions = []
    
    # Apply ALL models to the SAME patch
    for model_path, model_data in self.loaded_models.items():
        model = model_data['model']
        config = model_data['config']
        
        # Run inference for this model on this patch
        results = model(patch_info.image, 
                      conf=config.min_conf, 
                      iou=0.5,
                      max_det=100,
                      verbose=False)
        
        # Process results and add to all_predictions
        all_predictions.extend(processed_predictions)
    
    return all_predictions
```

### Performance Analysis

**Current Bottlenecks:**
- **Sequential model loading**: Each model loads separately for each patch
- **No batching**: Each patch processed individually
- **Memory inefficiency**: Models loaded multiple times
- **GPU underutilization**: Single patch doesn't utilize full GPU capacity

**Example with 3 models and 1000 patches:**
- **Current**: 3,000 model calls (1000 patches × 3 models)
- **Inefficient**: Each model call processes only 1 patch
- **GPU utilization**: ~10-20% (single patch doesn't fill GPU)

### Recommended Batch Processing Strategy

#### 1. **Batch-by-Model Approach**

```python
def process_wsi_batched(self, wsi_path: str, batch_size: int = 32) -> List[Prediction]:
    """Optimized batch processing"""
    all_predictions = []
    
    # Process each model separately with batching
    for model_path, model_data in self.loaded_models.items():
        model = model_data['model']
        
        # Create batches of patches for this model
        for batch_start in range(0, len(patches), batch_size):
            batch_patches = patches[batch_start:batch_start + batch_size]
            
            # Prepare batch tensor
            batch_images = torch.stack([patch.image for patch in batch_patches])
            
            # Single model call for entire batch
            results = model(batch_images, 
                          conf=model_data['config'].min_conf,
                          iou=0.3,
                          max_det=50,
                          verbose=False)
            
            # Process batch results
            batch_predictions = self._process_batch_results(results, batch_patches)
            all_predictions.extend(batch_predictions)
    
    return all_predictions
```

#### 2. **Multi-Model Batch Processing**

```python
def process_wsi_multi_model_batch(self, wsi_path: str, batch_size: int = 16) -> List[Prediction]:
    """Process multiple models in parallel batches"""
    all_predictions = []
    
    # Create batches across all models
    for batch_start in range(0, len(patches), batch_size):
        batch_patches = patches[batch_start:batch_start + batch_size]
        
        # Process all models for this batch
        batch_predictions = []
        for model_path, model_data in self.loaded_models.items():
            model = model_data['model']
            
            # Prepare batch for this model
            batch_images = self._prepare_batch_for_model(batch_patches, model_data)
            
            # Run inference
            results = model(batch_images, 
                          conf=model_data['config'].min_conf,
                          iou=0.3,
                          max_det=50)
            
            # Process results
            model_predictions = self._process_model_results(results, batch_patches, model_data)
            batch_predictions.extend(model_predictions)
        
        all_predictions.extend(batch_predictions)
    
    return all_predictions
```

### Performance Comparison

| Approach | Model Calls | GPU Utilization | Memory Usage | Speed |
|----------|-------------|-----------------|--------------|-------|
| **Current (Sequential)** | N × M | 10-20% | Low | 1x |
| **Batch-by-Model** | M × (N/B) | 60-80% | Medium | 3-5x |
| **Multi-Model Batch** | N/B | 80-95% | High | 5-10x |

Where: N = patches, M = models, B = batch_size

### Implementation Recommendations

#### 1. **Adaptive Batch Sizing**

```python
def calculate_optimal_batch_size(model, gpu_memory_gb: float) -> int:
    """Calculate optimal batch size based on GPU memory"""
    if gpu_memory_gb >= 24:
        return 32
    elif gpu_memory_gb >= 16:
        return 16
    elif gpu_memory_gb >= 8:
        return 8
    else:
        return 4
```

#### 2. **Memory Management**

```python
def process_with_memory_management(self, patches: List[PatchInfo], batch_size: int):
    """Process with automatic memory management"""
    for batch_start in range(0, len(patches), batch_size):
        batch_patches = patches[batch_start:batch_start + batch_size]
        
        try:
            # Process batch
            batch_predictions = self._process_batch(batch_patches)
            yield batch_predictions
            
        except torch.cuda.OutOfMemoryError:
            # Reduce batch size and retry
            smaller_batch_size = batch_size // 2
            if smaller_batch_size >= 1:
                for sub_batch in self._split_batch(batch_patches, smaller_batch_size):
                    yield self._process_batch(sub_batch)
            else:
                # Process individually if batch size is 1
                for patch in batch_patches:
                    yield self._process_single_patch(patch)
        
        finally:
            # Clear GPU memory
            torch.cuda.empty_cache()
```

#### 3. **Parallel Model Processing**

```python
import concurrent.futures
from threading import Lock

def process_models_parallel(self, batch_patches: List[PatchInfo]) -> List[Prediction]:
    """Process multiple models in parallel for a batch"""
    all_predictions = []
    prediction_lock = Lock()
    
    def process_model_batch(model_path, model_data):
        model = model_data['model']
        batch_images = self._prepare_batch(batch_patches, model_data)
        
        results = model(batch_images, 
                       conf=model_data['config'].min_conf,
                       iou=0.3,
                       max_det=50)
        
        predictions = self._process_model_results(results, batch_patches, model_data)
        
        with prediction_lock:
            all_predictions.extend(predictions)
    
    # Process all models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.loaded_models)) as executor:
        futures = []
        for model_path, model_data in self.loaded_models.items():
            future = executor.submit(process_model_batch, model_path, model_data)
            futures.append(future)
        
        # Wait for all models to complete
        concurrent.futures.wait(futures)
    
    return all_predictions
```

### Configuration for Batch Processing

```python
# Recommended configuration
BATCH_PROCESSING_CONFIG = {
    'batch_size': 16,              # Optimal for most GPUs
    'max_batch_size': 32,          # Maximum batch size
    'min_batch_size': 4,           # Minimum batch size
    'memory_threshold': 0.8,       # GPU memory usage threshold
    'parallel_models': True,       # Process models in parallel
    'adaptive_batching': True,     # Automatically adjust batch size
}
```

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

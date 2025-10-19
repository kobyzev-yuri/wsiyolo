"""
WSI YOLO - Система анализа Whole Slide Images с YOLO моделями
"""

from .data_structures import Coords, Box, Prediction, Model, PatchInfo, WSIInfo
from .monai_pipeline import WSIPipeline
from .yolo_inference import YOLOInference
from .polygon_merger import PolygonMerger
from .main import WSIYOLOPipeline, create_models_config, main

__version__ = "1.0.0"
__author__ = "WSI YOLO Team"

__all__ = [
    "Coords",
    "Box", 
    "Prediction",
    "Model",
    "PatchInfo",
    "WSIInfo",
    "WSIPipeline",
    "YOLOInference", 
    "PolygonMerger",
    "WSIYOLOPipeline",
    "create_models_config",
    "main"
]


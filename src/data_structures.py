"""
Базовые структуры данных для WSI YOLO проекта.
Соответствует спецификации WSIYOLO.md
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class Coords:
    """Координаты точки"""
    x: float
    y: float


@dataclass
class Box:
    """Прямоугольник с начальной и конечной точками"""
    start: Coords
    end: Coords

    def area(self) -> float:
        """Площадь прямоугольника"""
        width = self.end.x - self.start.x
        height = self.end.y - self.start.y
        return width * height

    def center(self) -> Coords:
        """Центр прямоугольника"""
        return Coords(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def intersection_area(self, box: "Box") -> float:
        """Площадь пересечения с другим прямоугольником"""
        inter_x1 = max(self.start.x, box.start.x)
        inter_y1 = max(self.start.y, box.start.y)
        inter_x2 = min(self.end.x, box.end.x)
        inter_y2 = min(self.end.y, box.end.y)
        return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    def intersects(self, other: "Box") -> bool:
        """Проверка пересечения с другим прямоугольником"""
        return self.intersection_area(other) > 0

    def iou(self, other: "Box") -> float:
        """Intersection over Union с другим прямоугольником"""
        inter_area = self.intersection_area(other)
        union_area = self.area() + other.area() - inter_area
        return inter_area / union_area if union_area > 0 else 0


@dataclass
class Prediction:
    """Предсказание модели"""
    class_name: str
    box: Box
    conf: float
    polygon: Optional[List[Coords]] = None


@dataclass
class Model:
    """Конфигурация модели"""
    model_path: str
    window_size: int
    min_conf: float = 0.5
    
    def __post_init__(self):
        """Валидация параметров модели"""
        if self.window_size <= 0:
            raise ValueError("window_size должен быть положительным")
        if not 0 <= self.min_conf <= 1:
            raise ValueError("min_conf должен быть между 0 и 1")


@dataclass
class PatchInfo:
    """Информация о патче"""
    patch_id: int
    x: int  # Абсолютная координата X в WSI
    y: int  # Абсолютная координата Y в WSI
    size: int  # Размер патча
    image: np.ndarray  # Изображение патча
    has_tissue: bool = True  # Содержит ли патч ткань


@dataclass
class WSIInfo:
    """Информация о WSI"""
    path: str
    width: int
    height: int
    levels: int
    level_downsamples: List[float]
    mpp: Optional[float] = None  # Микрометры на пиксель


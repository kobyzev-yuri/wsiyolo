class Box:
    start: Coords
    end: Coords

    def area(self) -> float:
        width = self.end.x - self.start.x
        height = self.end.y - self.start.y
        return width * height

    def center(self) -> Coords:
        return Coords(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def intersection_area(self, box: "Box") -> float:
        inter_x1 = max(self.start.x, box.start.x)
        inter_y1 = max(self.start.y, box.start.y)
        inter_x2 = min(self.end.x, box.end.x)
        inter_y2 = min(self.end.y, box.end.y)
        return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    def intersects(self, other: "Box") -> bool:
        return self.intersection_area(other) > 0

    def iou(self, other: "Box") -> float:
        inter_area = self.intersection_area(other)
        union_area = self.area() + other.area() - inter_area
        return inter_area / union_area if union_area > 0 else 0



def _to_union_polygons(
        self, preds: list[domain.Prediction]
    ) -> list[list[tuple[float, float]]]:
        pred_polygons = []
        for pred in preds:
            p = self._to_polygon(pred)
            if not p.is_valid:
                p = p.buffer(0)
                if not p.is_valid:
                    continue

            pred_polygons.append(p)

        pred_unioned_polygons = unary_union(pred_polygons)
        if pred_unioned_polygons is None:
            return []

        if pred_unioned_polygons.geom_type == "MultiPolygon":
            result = []
            for poly in pred_unioned_polygons.geoms:
                result.append(list(poly.exterior.coords))
            return result

        return [list(pred_unioned_polygons.exterior.coords)]



class Model
     model_path: str
     window_size: int
     min_conf: float



func(models: list[Model], wsi_path: str) -> list[Prediction]


class Coords:
    x: float
    y: float

class Box:
    start: Coords
    end: Coords

class Prediction:
    class_name: str
    box: Box
    conf: float
    polygon: Optional[list[Coords]]

class Model
     model_path: str
     window_size: int
     
     
     
import cv2
import numpy as np

def _is_background(img, threshold_value=200, background_ratio=0.99):
    """
    Определяет, является ли изображение в основном фоном (почти однотонным).

    :param img: входное изображение (numpy.ndarray)
    :param threshold_value: порог для бинаризации
    :param background_ratio: минимальная доля светлых пикселей, чтобы считать фон фоном
    :return: True, если изображение — в основном фон, иначе False
    """
    # Если изображение цветное, переводим в градации серого
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Бинаризация по порогу
    _, thresh_image = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Подсчёт количества "фоновых" пикселей
    background_pixels = np.sum(thresh_image == 255)
    total_pixels = img_gray.size

    # Доля фоновых пикселей
    background_ratio_actual = background_pixels / total_pixels

    return background_ratio_actual >= background_ratio     

def _is_background_hsv(img, saturation_threshold=30, tissue_ratio=0.1):
    """
    Альтернативный метод определения фона на основе HSV анализа.
    Более точный, но медленнее чем _is_background.
    
    :param img: входное изображение (numpy.ndarray)
    :param saturation_threshold: порог насыщенности для определения ткани
    :param tissue_ratio: минимальная доля ткани в патче
    :return: True, если патч содержит ткань, False если фон
    """
    # Конвертация в HSV
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        # Если уже grayscale, создаем 3-канальное изображение
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Анализ насыщенности (S канал)
    saturation = hsv[:, :, 1]
    
    # OTSU thresholding для автоматического порога
    _, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Морфологические операции для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Подсчет процента ткани
    tissue_pixels = np.sum(binary > 0)
    total_pixels = binary.size
    tissue_ratio_actual = tissue_pixels / total_pixels
    
    # Возвращаем True если есть достаточно ткани (НЕ фон)
    return tissue_ratio_actual >= tissue_ratio




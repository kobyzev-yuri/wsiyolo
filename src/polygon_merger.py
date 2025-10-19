"""
Объединение перекрывающихся полигонов.
Использует shapely для геометрических операций.
"""

from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from data_structures import Prediction, Coords, Box


class PolygonMerger:
    """Класс для объединения перекрывающихся полигонов"""
    
    def __init__(self, iou_threshold: float = 0.5, min_area: float = 10.0):
        """
        Инициализация
        
        Args:
            iou_threshold: Порог IoU для объединения полигонов
            min_area: Минимальная площадь полигона
        """
        self.iou_threshold = iou_threshold
        self.min_area = min_area
    
    def merge_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Объединяет перекрывающиеся предсказания
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            List[Prediction]: Объединенные предсказания
        """
        if not predictions:
            return []
        
        # Группируем предсказания по классам
        grouped_predictions = self._group_by_class(predictions)
        
        merged_predictions = []
        
        for class_name, class_predictions in grouped_predictions.items():
            # Объединяем предсказания одного класса
            merged_class_predictions = self._merge_class_predictions(class_predictions)
            merged_predictions.extend(merged_class_predictions)
        
        return merged_predictions
    
    def _group_by_class(self, predictions: List[Prediction]) -> dict:
        """
        Группирует предсказания по классам
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            dict: Словарь {class_name: [predictions]}
        """
        grouped = {}
        for pred in predictions:
            if pred.class_name not in grouped:
                grouped[pred.class_name] = []
            grouped[pred.class_name].append(pred)
        return grouped
    
    def _merge_class_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Объединяет предсказания одного класса
        
        Args:
            predictions: Список предсказаний одного класса
            
        Returns:
            List[Prediction]: Объединенные предсказания
        """
        if len(predictions) <= 1:
            return predictions
        
        print(f"   Объединение {len(predictions)} предсказаний класса {predictions[0].class_name}")
        
        # Создаем полигоны из предсказаний
        polygons = []
        
        for i, pred in enumerate(predictions):
            if pred.polygon:
                try:
                    # Создаем shapely полигон
                    coords = [(p.x, p.y) for p in pred.polygon]
                    if len(coords) >= 3:  # Минимум 3 точки для полигона
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polygons.append(poly)
                        else:
                            print(f"⚠️  Невалидный полигон для предсказания {i}: {pred.class_name}")
                    else:
                        print(f"⚠️  Недостаточно точек для полигона {i}: {len(coords)}")
                except Exception as e:
                    print(f"⚠️  Ошибка создания полигона {i} (класс: {pred.class_name}): {e}")
                    print(f"   Количество точек: {len(pred.polygon) if pred.polygon else 0}")
                    print(f"   Тип ошибки: {type(e).__name__}")
                    import traceback
                    print(f"   Стек ошибки: {traceback.format_exc()}")
                    continue
        
        if not polygons:
            return predictions
        
        # Объединяем полигоны
        try:
            merged_polygons = unary_union(polygons)
            
            # Обрабатываем результат объединения
            if merged_polygons.is_empty:
                return predictions
            
            # Создаем новые предсказания из объединенных полигонов
            merged_predictions = []
            
            # Получаем класс из первого предсказания (все должны быть одного класса)
            class_name = predictions[0].class_name if predictions else "unknown"
            
            if isinstance(merged_polygons, MultiPolygon):
                for poly in merged_polygons.geoms:
                    if poly.area >= self.min_area:
                        merged_pred = self._polygon_to_prediction(poly, class_name)
                        if merged_pred:
                            merged_predictions.append(merged_pred)
            else:
                if merged_polygons.area >= self.min_area:
                    merged_pred = self._polygon_to_prediction(merged_polygons, class_name)
                    if merged_pred:
                        merged_predictions.append(merged_pred)
            
            return merged_predictions if merged_predictions else predictions
            
        except Exception as e:
            print(f"⚠️  Ошибка объединения полигонов: {e}")
            return predictions
    
    def _polygon_to_prediction(self, polygon: Polygon, class_name: str) -> Prediction:
        """
        Преобразует shapely полигон в Prediction с упрощением
        
        Args:
            polygon: Shapely полигон
            class_name: Название класса
            
        Returns:
            Prediction: Предсказание
        """
        try:
            # Умное упрощение до максимум 60 точек
            if len(polygon.exterior.coords) > 60:
                print(f"   Умное упрощение объединенного полигона: {len(polygon.exterior.coords)} точек")
                polygon = self._smart_simplify_polygon(polygon, max_points=60)
                print(f"   После умного упрощения: {len(polygon.exterior.coords)} точек")
            
            # Получаем границы полигона
            bounds = polygon.bounds
            minx, miny, maxx, maxy = bounds
            
            # Создаем bounding box
            box = Box(
                start=Coords(x=minx, y=miny),
                end=Coords(x=maxx, y=maxy)
            )
            
            # Создаем полигон из координат (исключаем последнюю дублирующуюся точку)
            coords = list(polygon.exterior.coords[:-1])
            polygon_coords = [Coords(x=x, y=y) for x, y in coords]
            
            # Вычисляем среднюю уверенность (можно улучшить)
            confidence = 0.8  # По умолчанию
            
            return Prediction(
                class_name=class_name,
                box=box,
                conf=confidence,
                polygon=polygon_coords
            )
            
        except Exception as e:
            print(f"⚠️  Ошибка преобразования полигона: {e}")
            return None
    
    def filter_by_iou(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Фильтрует предсказания по IoU, удаляя дубликаты
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            List[Prediction]: Отфильтрованные предсказания
        """
        if len(predictions) <= 1:
            return predictions
        
        # Сортируем по уверенности (убывание)
        sorted_predictions = sorted(predictions, key=lambda x: x.conf, reverse=True)
        
        filtered = []
        
        for pred in sorted_predictions:
            # Проверяем IoU с уже отфильтрованными предсказаниями
            is_duplicate = False
            
            for filtered_pred in filtered:
                if pred.class_name == filtered_pred.class_name:
                    iou = pred.box.iou(filtered_pred.box)
                    if iou > self.iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(pred)
        
        return filtered
    
    def merge_overlapping_boxes(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Объединяет перекрывающиеся bounding boxes
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            List[Prediction]: Объединенные предсказания
        """
        if len(predictions) <= 1:
            return predictions
        
        # Группируем по классам
        grouped = self._group_by_class(predictions)
        merged_predictions = []
        
        for class_name, class_predictions in grouped.items():
            # Сортируем по уверенности
            sorted_preds = sorted(class_predictions, key=lambda x: x.conf, reverse=True)
            
            merged_class_preds = []
            
            for pred in sorted_preds:
                # Проверяем пересечение с уже объединенными
                should_merge = False
                merge_with = None
                
                for merged_pred in merged_class_preds:
                    if pred.box.intersects(merged_pred.box):
                        iou = pred.box.iou(merged_pred.box)
                        if iou > self.iou_threshold:
                            should_merge = True
                            merge_with = merged_pred
                            break
                
                if should_merge and merge_with:
                    # Объединяем с существующим предсказанием
                    merged_box = self._merge_boxes(pred.box, merge_with.box)
                    merged_conf = max(pred.conf, merge_with.conf)
                    
                    # Обновляем существующее предсказание
                    merge_with.box = merged_box
                    merge_with.conf = merged_conf
                else:
                    # Добавляем новое предсказание
                    merged_class_preds.append(pred)
            
            merged_predictions.extend(merged_class_preds)
        
        return merged_predictions
    
    def _merge_boxes(self, box1: Box, box2: Box) -> Box:
        """
        Объединяет два bounding box
        
        Args:
            box1: Первый box
            box2: Второй box
            
        Returns:
            Box: Объединенный box
        """
        min_x = min(box1.start.x, box2.start.x)
        min_y = min(box1.start.y, box2.start.y)
        max_x = max(box1.end.x, box2.end.x)
        max_y = max(box1.end.y, box2.end.y)
        
        return Box(
            start=Coords(x=min_x, y=min_y),
            end=Coords(x=max_x, y=max_y)
        )
    
    def _smart_simplify_polygon(self, polygon: Polygon, max_points: int = 60) -> Polygon:
        """
        Умное упрощение полигона до заданного количества точек
        
        Args:
            polygon: Исходный полигон
            max_points: Максимальное количество точек
            
        Returns:
            Polygon: Упрощенный полигон
        """
        try:
            current_poly = polygon
            current_points = len(current_poly.exterior.coords)
            
            if current_points <= max_points:
                return current_poly
            
            # Адаптивное упрощение с бинарным поиском tolerance
            min_tolerance = 0.1
            max_tolerance = 10.0
            best_poly = current_poly
            
            # Бинарный поиск оптимального tolerance
            for _ in range(10):  # Максимум 10 итераций
                tolerance = (min_tolerance + max_tolerance) / 2
                simplified = current_poly.simplify(tolerance, preserve_topology=True)
                
                if simplified.is_valid and len(simplified.exterior.coords) > 3:
                    points_count = len(simplified.exterior.coords)
                    
                    if points_count <= max_points:
                        # Нашли подходящий результат
                        best_poly = simplified
                        min_tolerance = tolerance
                        if points_count >= max_points * 0.8:  # Если достаточно близко к цели
                            break
                    else:
                        # Нужно больше упрощения
                        max_tolerance = tolerance
                else:
                    # Упрощение слишком агрессивное
                    max_tolerance = tolerance
            
            # Если все еще слишком много точек, используем равномерную выборку
            if len(best_poly.exterior.coords) > max_points:
                coords = list(best_poly.exterior.coords)
                step = len(coords) // max_points
                sampled_coords = coords[::max(1, step)]
                
                # Создаем новый полигон из выбранных точек
                if len(sampled_coords) >= 3:
                    sampled_poly = Polygon(sampled_coords)
                    if sampled_poly.is_valid:
                        best_poly = sampled_poly
            
            return best_poly
            
        except Exception as e:
            print(f"⚠️  Ошибка умного упрощения: {e}")
            return polygon


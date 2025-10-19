"""
YOLO11-seg инференс для патчей WSI.
Извлекает метки классов из моделей и выполняет предсказания.
"""

import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import torch
from shapely.geometry import Polygon

from .data_structures import PatchInfo, Prediction, Model, Coords, Box


class YOLOInference:
    """Класс для выполнения YOLO инференса на патчах"""
    
    def __init__(self, models: List[Model]):
        """
        Инициализация с моделями
        
        Args:
            models: Список конфигураций моделей
        """
        self.models = models
        self.loaded_models = {}
        
        # Загружаем все модели
        for model_config in models:
            self._load_model(model_config)
    
    def _load_model(self, model_config: Model):
        """
        Загружает YOLO модель
        
        Args:
            model_config: Конфигурация модели
        """
        try:
            model = YOLO(model_config.model_path)
            
            # Извлекаем метки классов из модели
            class_names = model.names
            
            # Сохраняем модель с метаданными
            self.loaded_models[model_config.model_path] = {
                'model': model,
                'config': model_config,
                'class_names': class_names,
                'num_classes': len(class_names)
            }
            
            print(f"✅ Загружена модель: {model_config.model_path}")
            print(f"   Классы: {list(class_names.values())}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {model_config.model_path}: {e}")
            raise
    
    def predict_patch(self, patch_info: PatchInfo) -> List[Prediction]:
        """
        Выполняет предсказания на одном патче для всех моделей
        
        Args:
            patch_info: Информация о патче
            
        Returns:
            List[Prediction]: Список предсказаний
        """
        all_predictions = []
        
        for model_path, model_data in self.loaded_models.items():
            model = model_data['model']
            config = model_data['config']
            class_names = model_data['class_names']
            
            # Проверяем размер патча
            if patch_info.size != config.window_size:
                print(f"⚠️  Размер патча {patch_info.size} не соответствует размеру модели {config.window_size}")
                continue
            
            try:
                # Выполняем предсказание с NMS параметрами
                results = model(patch_info.image, 
                              conf=config.min_conf, 
                              iou=0.5,  # NMS IoU threshold
                              max_det=100,  # Максимум детекций
                              verbose=False)
                
                # Обрабатываем результаты
                for result in results:
                    if result.masks is not None:
                        # Обрабатываем маски сегментации
                        predictions = self._process_segmentation_results(
                            result, patch_info, class_names
                        )
                        all_predictions.extend(predictions)
                    else:
                        # Обрабатываем только bounding boxes
                        predictions = self._process_detection_results(
                            result, patch_info, class_names
                        )
                        all_predictions.extend(predictions)
                        
            except Exception as e:
                print(f"❌ Ошибка предсказания для модели {model_path}: {e}")
                continue
        
        return all_predictions
    
    def _process_segmentation_results(self, result, patch_info: PatchInfo, class_names: Dict) -> List[Prediction]:
        """
        Обрабатывает результаты сегментации
        
        Args:
            result: Результат YOLO
            patch_info: Информация о патче
            class_names: Словарь классов
            
        Returns:
            List[Prediction]: Список предсказаний
        """
        predictions = []
        
        if result.masks is None:
            return predictions
        
        # Получаем маски и bounding boxes
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i, (mask, box, conf, class_id) in enumerate(zip(masks, boxes, confidences, class_ids)):
            # Получаем название класса
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            # Преобразуем координаты в абсолютные
            x1, y1, x2, y2 = box
            absolute_box = Box(
                start=Coords(x=patch_info.x + x1, y=patch_info.y + y1),
                end=Coords(x=patch_info.x + x2, y=patch_info.y + y2)
            )
            
            # Создаем полигон из маски
            polygon = self._mask_to_polygon(mask, patch_info.x, patch_info.y)
            
            # Фильтруем по минимальному размеру bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            min_size = 20  # Минимальный размер 20x20 пикселей
            
            if bbox_width >= min_size and bbox_height >= min_size:
                # Создаем предсказание
                prediction = Prediction(
                    class_name=class_name,
                    box=absolute_box,
                    conf=float(conf),
                    polygon=polygon
                )
                
                predictions.append(prediction)
            else:
                print(f"   Фильтрован мелкий bbox: {bbox_width:.1f}x{bbox_height:.1f} для {class_name}")
        
        return predictions
    
    def _process_detection_results(self, result, patch_info: PatchInfo, class_names: Dict) -> List[Prediction]:
        """
        Обрабатывает результаты детекции (только bounding boxes)
        
        Args:
            result: Результат YOLO
            patch_info: Информация о патче
            class_names: Словарь классов
            
        Returns:
            List[Prediction]: Список предсказаний
        """
        predictions = []
        
        if result.boxes is None:
            return predictions
        
        # Получаем bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Получаем название класса
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            # Преобразуем координаты в абсолютные
            x1, y1, x2, y2 = box
            absolute_box = Box(
                start=Coords(x=patch_info.x + x1, y=patch_info.y + y1),
                end=Coords(x=patch_info.x + x2, y=patch_info.y + y2)
            )
            
            # Фильтруем по минимальному размеру bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            min_size = 20  # Минимальный размер 20x20 пикселей
            
            if bbox_width >= min_size and bbox_height >= min_size:
                # Создаем предсказание без полигона
                prediction = Prediction(
                    class_name=class_name,
                    box=absolute_box,
                    conf=float(conf),
                    polygon=None
                )
                
                predictions.append(prediction)
            else:
                print(f"   Фильтрован мелкий bbox: {bbox_width:.1f}x{bbox_height:.1f} для {class_name}")
        
        return predictions
    
    def _mask_to_polygon(self, mask: np.ndarray, offset_x: int, offset_y: int) -> List[Coords]:
        """
        Преобразует маску в полигон с упрощением
        
        Args:
            mask: Маска сегментации
            offset_x: Смещение по X
            offset_y: Смещение по Y
            
        Returns:
            List[Coords]: Полигон в абсолютных координатах
        """
        try:
            from skimage import measure
            
            # Находим контуры
            contours = measure.find_contours(mask, 0.5)
            
            if not contours:
                return []
            
            # Берем самый большой контур
            largest_contour = max(contours, key=len)
            
            # Преобразуем в абсолютные координаты
            coords = []
            for point in largest_contour:
                x, y = point[1], point[0]  # skimage использует (row, col)
                coords.append((offset_x + x, offset_y + y))
            
            # Создаем Shapely полигон для умного упрощения
            if len(coords) >= 3:
                try:
                    poly = Polygon(coords)
                    if poly.is_valid:
                        # Умное упрощение до максимум 60 точек
                        simplified = self._smart_simplify_polygon(poly, max_points=60)
                        
                        # Преобразуем обратно в Coords
                        polygon = []
                        for x, y in simplified.exterior.coords[:-1]:  # Исключаем последнюю дублирующуюся точку
                            polygon.append(Coords(x=x, y=y))
                        
                        print(f"   Умное упрощение полигона: {len(coords)} -> {len(polygon)} точек")
                        return polygon
                    else:
                        print(f"⚠️  Невалидный полигон, используем исходный")
                        return [Coords(x=x, y=y) for x, y in coords]
                except Exception as e:
                    print(f"⚠️  Ошибка упрощения полигона: {e}, используем исходный")
                    return [Coords(x=x, y=y) for x, y in coords]
            else:
                return []
            
        except ImportError:
            print("⚠️  scikit-image не установлен, полигоны не будут созданы")
            return []
        except Exception as e:
            print(f"⚠️  Ошибка создания полигона: {e}")
            return []
    
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
    
    def get_model_info(self) -> Dict:
        """
        Возвращает информацию о загруженных моделях
        
        Returns:
            Dict: Информация о моделях
        """
        info = {}
        for model_path, model_data in self.loaded_models.items():
            info[model_path] = {
                'class_names': model_data['class_names'],
                'num_classes': model_data['num_classes'],
                'window_size': model_data['config'].window_size,
                'min_conf': model_data['config'].min_conf
            }
        return info


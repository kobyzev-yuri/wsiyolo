"""
Улучшенный алгоритм объединения перекрывающихся полигонов.
Включает фильтрацию вложенных объектов, исключение background класса,
и фильтрацию коротких сегментов для lp модели.
"""

from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from data_structures import Prediction, Coords, Box


class ImprovedPolygonMerger:
    """Улучшенный класс для объединения перекрывающихся полигонов"""
    
    def __init__(self, 
                 iou_threshold: float = 0.7,  # Увеличен с 0.5 до 0.7
                 min_area: float = 50.0,      # Увеличен с 10.0 до 50.0
                 min_polygon_points: int = 8,  # Минимум точек для lp класса
                 lp_class_name: str = "lp",    # Название lp класса
                 background_class: str = "background"):  # Название background класса
        """
        Инициализация улучшенного merger
        
        Args:
            iou_threshold: Порог IoU для объединения (увеличен до 0.7)
            min_area: Минимальная площадь полигона (увеличена до 50.0)
            min_polygon_points: Минимум точек для lp полигонов
            lp_class_name: Название lp класса
            background_class: Название background класса
        """
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        self.min_polygon_points = min_polygon_points
        self.lp_class_name = lp_class_name
        self.background_class = background_class
    
    def merge_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Объединяет перекрывающиеся предсказания с улучшенной фильтрацией
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            List[Prediction]: Объединенные предсказания
        """
        if not predictions:
            return []
        
        print(f"🔧 Улучшенное объединение {len(predictions)} предсказаний...")
        
        # 1. Фильтрация background класса для lp модели
        filtered_predictions = self._filter_background_class(predictions)
        print(f"   После фильтрации background: {len(filtered_predictions)}")
        
        # 2. Фильтрация коротких сегментов для lp класса
        filtered_predictions = self._filter_short_segments(filtered_predictions)
        print(f"   После фильтрации коротких сегментов: {len(filtered_predictions)}")
        
        # 3. Группировка по классам
        grouped_predictions = self._group_by_class(filtered_predictions)
        
        merged_predictions = []
        
        for class_name, class_predictions in grouped_predictions.items():
            print(f"   Обработка класса {class_name}: {len(class_predictions)} предсказаний")
            
            # 4. Фильтрация вложенных объектов для lp класса
            if class_name == self.lp_class_name:
                class_predictions = self._filter_nested_objects(class_predictions)
                print(f"   После фильтрации вложенных объектов: {len(class_predictions)}")
            
            # 5. Объединение предсказаний класса
            merged_class_predictions = self._merge_class_predictions(class_predictions)
            merged_predictions.extend(merged_class_predictions)
        
        print(f"   Финальный результат: {len(merged_predictions)} предсказаний")
        return merged_predictions
    
    def _filter_background_class(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Исключает background класс из предсказаний lp модели
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            List[Prediction]: Отфильтрованные предсказания
        """
        filtered = []
        
        for pred in predictions:
            # Исключаем background класс
            if pred.class_name.lower() == self.background_class.lower():
                print(f"   Исключен background класс: {pred.class_name}")
                continue
            
            filtered.append(pred)
        
        return filtered
    
    def _filter_short_segments(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Фильтрует короткие сегменты для lp класса
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            List[Prediction]: Отфильтрованные предсказания
        """
        filtered = []
        
        for pred in predictions:
            # Проверяем только lp класс
            if pred.class_name == self.lp_class_name and pred.polygon:
                polygon_points = len(pred.polygon)
                
                # Фильтруем короткие сегменты
                if polygon_points < self.min_polygon_points:
                    print(f"   Исключен короткий сегмент lp: {polygon_points} точек")
                    continue
            
            filtered.append(pred)
        
        return filtered
    
    def _filter_nested_objects(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Фильтрует вложенные объекты для lp класса
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            List[Prediction]: Отфильтрованные предсказания
        """
        if len(predictions) <= 1:
            return predictions
        
        # Создаем полигоны для проверки вложенности
        polygons_with_predictions = []
        
        for pred in predictions:
            if pred.polygon and len(pred.polygon) >= 3:
                try:
                    coords = [(p.x, p.y) for p in pred.polygon]
                    poly = Polygon(coords)
                    if poly.is_valid:
                        polygons_with_predictions.append((poly, pred))
                except Exception as e:
                    print(f"   ⚠️  Ошибка создания полигона: {e}")
                    continue
        
        if not polygons_with_predictions:
            return predictions
        
        # Сортируем по площади (от больших к маленьким)
        polygons_with_predictions.sort(key=lambda x: x[0].area, reverse=True)
        
        # Проверяем вложенность - оставляем только самые большие объекты
        filtered_predictions = []
        
        for i, (poly1, pred1) in enumerate(polygons_with_predictions):
            is_nested = False
            
            # Проверяем только с уже отфильтрованными (большими) объектами
            for j, pred2 in enumerate(filtered_predictions):
                # Создаем полигон для pred2
                if pred2.polygon:
                    coords2 = [(p.x, p.y) for p in pred2.polygon]
                    poly2 = Polygon(coords2)
                else:
                    continue
                # Проверяем различные типы вложенности
                within_check = poly1.within(poly2)
                contains_check = poly2.contains(poly1)
                intersection_ratio = poly1.intersection(poly2).area / poly1.area if poly1.area > 0 else 0
                
                print(f"   Проверка вложенности: poly1({poly1.area:.1f}) vs poly2({poly2.area:.1f})")
                print(f"     within: {within_check}, contains: {contains_check}, intersection_ratio: {intersection_ratio:.3f}")
                
                # Объект считается вложенным если:
                # 1. Он полностью внутри другого (within)
                # 2. Другой объект содержит его (contains) 
                # 3. Большая часть его площади пересекается с другим объектом (>80%)
                if within_check or contains_check or intersection_ratio > 0.8:
                    print(f"   Исключен вложенный объект lp: площадь {poly1.area:.1f} вложена в {poly2.area:.1f}")
                    is_nested = True
                    break
            
            if not is_nested:
                filtered_predictions.append(pred1)
                print(f"   Сохранен объект lp: площадь {poly1.area:.1f}")
        
        return filtered_predictions
    
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
        Объединяет предсказания одного класса с улучшенной фильтрацией
        
        Args:
            predictions: Список предсказаний одного класса
            
        Returns:
            List[Prediction]: Объединенные предсказания
        """
        if len(predictions) <= 1:
            return predictions
        
        # Создаем полигоны из предсказаний
        polygons = []
        
        for i, pred in enumerate(predictions):
            if pred.polygon:
                try:
                    coords = [(p.x, p.y) for p in pred.polygon]
                    if len(coords) >= 3:
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polygons.append(poly)
                        else:
                            print(f"   ⚠️  Невалидный полигон для предсказания {i}")
                    else:
                        print(f"   ⚠️  Недостаточно точек для полигона {i}: {len(coords)}")
                except Exception as e:
                    print(f"   ⚠️  Ошибка создания полигона {i}: {e}")
                    continue
        
        if not polygons:
            return predictions
        
        # Объединяем полигоны
        try:
            merged_polygons = unary_union(polygons)
            
            if merged_polygons.is_empty:
                return predictions
            
            # Создаем новые предсказания из объединенных полигонов
            merged_predictions = []
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
            print(f"   ⚠️  Ошибка объединения полигонов: {e}")
            return predictions
    
    def _polygon_to_prediction(self, polygon: Polygon, class_name: str) -> Optional[Prediction]:
        """
        Преобразует shapely полигон в Prediction с улучшенным упрощением
        
        Args:
            polygon: Shapely полигон
            class_name: Название класса
            
        Returns:
            Optional[Prediction]: Предсказание или None
        """
        try:
            # Улучшенное упрощение для lp класса
            if class_name == self.lp_class_name:
                max_points = 60
            else:
                max_points = 40  # Меньше точек для других классов
            
            if len(polygon.exterior.coords) > max_points:
                print(f"   Упрощение полигона {class_name}: {len(polygon.exterior.coords)} -> {max_points} точек")
                polygon = self._smart_simplify_polygon(polygon, max_points=max_points)
            
            # Получаем границы полигона
            bounds = polygon.bounds
            minx, miny, maxx, maxy = bounds
            
            # Создаем bounding box
            box = Box(
                start=Coords(x=minx, y=miny),
                end=Coords(x=maxx, y=maxy)
            )
            
            # Создаем полигон из координат
            coords = list(polygon.exterior.coords[:-1])
            polygon_coords = [Coords(x=x, y=y) for x, y in coords]
            
            # Вычисляем среднюю уверенность
            confidence = 0.8  # Можно улучшить на основе исходных предсказаний
            
            return Prediction(
                class_name=class_name,
                box=box,
                conf=confidence,
                polygon=polygon_coords
            )
            
        except Exception as e:
            print(f"   ⚠️  Ошибка преобразования полигона: {e}")
            return None
    
    def _smart_simplify_polygon(self, polygon: Polygon, max_points: int = 60) -> Polygon:
        """
        Умное упрощение полигона с улучшенными параметрами
        
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
            
            # Улучшенные параметры упрощения
            min_tolerance = 0.2  # Увеличен с 0.1
            max_tolerance = 5.0   # Уменьшен с 10.0
            best_poly = current_poly
            
            # Бинарный поиск оптимального tolerance
            for _ in range(8):  # Уменьшено количество итераций
                tolerance = (min_tolerance + max_tolerance) / 2
                simplified = current_poly.simplify(tolerance, preserve_topology=True)
                
                if simplified.is_valid and len(simplified.exterior.coords) > 3:
                    points_count = len(simplified.exterior.coords)
                    
                    if points_count <= max_points:
                        best_poly = simplified
                        min_tolerance = tolerance
                        if points_count >= max_points * 0.9:  # Более строгое условие
                            break
                    else:
                        max_tolerance = tolerance
                else:
                    max_tolerance = tolerance
            
            # Если все еще слишком много точек, используем равномерную выборку
            if len(best_poly.exterior.coords) > max_points:
                coords = list(best_poly.exterior.coords)
                step = len(coords) // max_points
                sampled_coords = coords[::max(1, step)]
                
                if len(sampled_coords) >= 3:
                    sampled_poly = Polygon(sampled_coords)
                    if sampled_poly.is_valid:
                        best_poly = sampled_poly
            
            return best_poly
            
        except Exception as e:
            print(f"   ⚠️  Ошибка умного упрощения: {e}")
            return polygon
    
    def filter_by_improved_iou(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Фильтрует предсказания по улучшенному IoU (threshold=0.7)
        
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
            is_duplicate = False
            
            for filtered_pred in filtered:
                if pred.class_name == filtered_pred.class_name:
                    iou = pred.box.iou(filtered_pred.box)
                    if iou > self.iou_threshold:  # 0.7 вместо 0.5
                        is_duplicate = True
                        print(f"   Исключен дубликат по IoU {iou:.3f} > {self.iou_threshold}")
                        break
            
            if not is_duplicate:
                filtered.append(pred)
        
        return filtered
    
    def get_filtering_statistics(self, original_predictions: List[Prediction], 
                                filtered_predictions: List[Prediction]) -> dict:
        """
        Возвращает статистику фильтрации
        
        Args:
            original_predictions: Исходные предсказания
            filtered_predictions: Отфильтрованные предсказания
            
        Returns:
            dict: Статистика фильтрации
        """
        total_original = len(original_predictions)
        total_filtered = len(filtered_predictions)
        filtered_out = total_original - total_filtered
        
        # Статистика по классам
        class_stats = {}
        for pred in original_predictions:
            class_name = pred.class_name
            if class_name not in class_stats:
                class_stats[class_name] = {'original': 0, 'filtered': 0}
            class_stats[class_name]['original'] += 1
        
        for pred in filtered_predictions:
            class_name = pred.class_name
            if class_name in class_stats:
                class_stats[class_name]['filtered'] += 1
        
        return {
            'total_original': total_original,
            'total_filtered': total_filtered,
            'filtered_out': filtered_out,
            'filtering_ratio': filtered_out / total_original if total_original > 0 else 0,
            'class_statistics': class_stats
        }

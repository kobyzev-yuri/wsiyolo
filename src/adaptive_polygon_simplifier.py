#!/usr/bin/env python3
"""
Адаптивный алгоритм упрощения полигонов.
Фокусируется на сохранении точности площади и удалении избыточных точек на прямых линиях.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional
import math

class AdaptivePolygonSimplifier:
    """Адаптивный упроститель полигонов с фокусом на точность площади"""
    
    def __init__(self):
        # Параметры для разных типов полигонов
        self.simple_threshold = 20      # Полигоны с <20 точек считаются простыми
        self.complex_threshold = 200    # Полигоны с >200 точек считаются сложными
        
        # Адаптивные параметры упрощения
        self.simple_params = {
            'min_tolerance': 0.05,
            'max_tolerance': 1.0,
            'max_points': 15,
            'area_preservation_threshold': 0.95  # Сохранить 95% площади
        }
        
        self.complex_params = {
            'min_tolerance': 0.1,
            'max_tolerance': 3.0,
            'max_points': 80,
            'area_preservation_threshold': 0.98  # Сохранить 98% площади
        }
        
        self.default_params = {
            'min_tolerance': 0.1,
            'max_tolerance': 2.0,
            'max_points': 60,
            'area_preservation_threshold': 0.97  # Сохранить 97% площади
        }
    
    def simplify_polygon(self, polygon: Polygon, target_points: Optional[int] = None) -> Tuple[Polygon, dict]:
        """
        Адаптивное упрощение полигона с сохранением точности площади
        
        Args:
            polygon: Исходный полигон
            target_points: Целевое количество точек (None для автоматического выбора)
            
        Returns:
            Tuple[Polygon, dict]: (упрощенный полигон, метрики качества)
        """
        if not polygon.is_valid or polygon.is_empty:
            return polygon, {'error': 'Invalid or empty polygon'}
        
        original_points = len(polygon.exterior.coords)
        original_area = polygon.area
        original_perimeter = polygon.length
        
        # Определяем параметры в зависимости от сложности полигона
        if original_points < self.simple_threshold:
            params = self.simple_params
            complexity = 'simple'
        elif original_points > self.complex_threshold:
            params = self.complex_params
            complexity = 'complex'
        else:
            params = self.default_params
            complexity = 'medium'
        
        # Если не указано целевое количество точек, используем параметры
        if target_points is None:
            target_points = params['max_points']
        
        # Если полигон уже достаточно простой, возвращаем как есть
        if original_points <= target_points:
            return polygon, {
                'original_points': original_points,
                'simplified_points': original_points,
                'area_preserved': 1.0,
                'perimeter_preserved': 1.0,
                'complexity': complexity,
                'method': 'no_simplification_needed'
            }
        
        # Адаптивное упрощение с проверкой качества
        best_polygon, best_metrics = self._adaptive_simplify(
            polygon, target_points, params, complexity
        )
        
        # Вычисляем финальные метрики
        final_metrics = {
            'original_points': original_points,
            'simplified_points': len(best_polygon.exterior.coords),
            'area_preserved': best_polygon.area / original_area if original_area > 0 else 0,
            'perimeter_preserved': best_polygon.length / original_perimeter if original_perimeter > 0 else 0,
            'complexity': complexity,
            'method': best_metrics.get('method', 'adaptive_simplify'),
            'tolerance_used': best_metrics.get('tolerance_used', 0),
            'iterations': best_metrics.get('iterations', 0)
        }
        
        return best_polygon, final_metrics
    
    def _adaptive_simplify(self, polygon: Polygon, target_points: int, params: dict, complexity: str) -> Tuple[Polygon, dict]:
        """Адаптивное упрощение с проверкой качества"""
        
        min_tolerance = params['min_tolerance']
        max_tolerance = params['max_tolerance']
        area_threshold = params['area_preservation_threshold']
        
        best_polygon = polygon
        best_score = 0
        best_metrics = {}
        
        # Бинарный поиск оптимального tolerance
        for iteration in range(12):  # Увеличиваем количество итераций
            tolerance = (min_tolerance + max_tolerance) / 2
            
            try:
                simplified = polygon.simplify(tolerance, preserve_topology=True)
                
                if not simplified.is_valid or len(simplified.exterior.coords) < 3:
                    # Упрощение слишком агрессивное
                    max_tolerance = tolerance
                    continue
                
                # Проверяем качество упрощения
                area_preserved = simplified.area / polygon.area if polygon.area > 0 else 0
                points_count = len(simplified.exterior.coords)
                
                # Вычисляем оценку качества
                if area_preserved >= area_threshold:
                    # Площадь сохранена хорошо
                    if points_count <= target_points:
                        # Количество точек подходящее
                        score = area_preserved * (target_points / max(points_count, 1))
                        if score > best_score:
                            best_score = score
                            best_polygon = simplified
                            best_metrics = {
                                'method': 'douglas_peucker',
                                'tolerance_used': tolerance,
                                'iterations': iteration + 1,
                                'area_preserved': area_preserved,
                                'points_reduction': (len(polygon.exterior.coords) - points_count) / len(polygon.exterior.coords)
                            }
                        min_tolerance = tolerance
                    else:
                        # Нужно больше упрощения
                        max_tolerance = tolerance
                else:
                    # Площадь потеряна, нужно менее агрессивное упрощение
                    max_tolerance = tolerance
                    
            except Exception as e:
                max_tolerance = tolerance
                continue
        
        # Если не удалось найти хорошее упрощение, пробуем альтернативные методы
        if best_score == 0 or len(best_polygon.exterior.coords) > target_points * 1.5:
            best_polygon, best_metrics = self._fallback_simplify(polygon, target_points, params)
        
        return best_polygon, best_metrics
    
    def _fallback_simplify(self, polygon: Polygon, target_points: int, params: dict) -> Tuple[Polygon, dict]:
        """Альтернативные методы упрощения"""
        
        try:
            # Метод 1: Равномерная выборка с сохранением ключевых точек
            coords = list(polygon.exterior.coords)
            if len(coords) <= target_points:
                return polygon, {'method': 'no_simplification_needed'}
            
            # Находим ключевые точки (углы, высокие кривизны)
            key_points = self._find_key_points(coords)
            
            # Если ключевых точек достаточно, используем их
            if len(key_points) <= target_points:
                key_coords = [coords[i] for i in key_points]
                try:
                    simplified = Polygon(key_coords)
                    if simplified.is_valid:
                        return simplified, {
                            'method': 'key_points_preservation',
                            'key_points_count': len(key_points)
                        }
                except:
                    pass
            
            # Метод 2: Адаптивная выборка
            step = len(coords) / target_points
            sampled_coords = []
            
            for i in range(target_points):
                idx = int(i * step) % len(coords)
                sampled_coords.append(coords[idx])
            
            # Замыкаем полигон
            if sampled_coords[0] != sampled_coords[-1]:
                sampled_coords.append(sampled_coords[0])
            
            simplified = Polygon(sampled_coords)
            if simplified.is_valid:
                return simplified, {
                    'method': 'adaptive_sampling',
                    'sampling_step': step
                }
            
            # Если ничего не работает, возвращаем исходный
            return polygon, {'method': 'fallback_failed'}
            
        except Exception as e:
            return polygon, {'method': 'fallback_error', 'error': str(e)}
    
    def _find_key_points(self, coords: List[Tuple[float, float]], min_angle: float = 15.0) -> List[int]:
        """Находит ключевые точки (углы, высокие кривизны)"""
        if len(coords) < 3:
            return list(range(len(coords)))
        
        key_points = [0]  # Всегда включаем первую точку
        
        for i in range(1, len(coords) - 1):
            # Вычисляем угол в точке
            p1 = np.array(coords[i-1])
            p2 = np.array(coords[i])
            p3 = np.array(coords[i+1])
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Нормализуем векторы
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                v1_norm = v1 / norm1
                v2_norm = v2 / norm2
                
                # Вычисляем угол
                cos_angle = np.dot(v1_norm, v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = math.degrees(math.acos(cos_angle))
                
                # Если угол острый, это ключевая точка
                if angle < (180 - min_angle):
                    key_points.append(i)
        
        # Всегда включаем последнюю точку
        if len(coords) - 1 not in key_points:
            key_points.append(len(coords) - 1)
        
        return key_points
    
    def calculate_polygon_metrics(self, polygon: Polygon) -> dict:
        """Вычисляет метрики полигона для анализа качества"""
        if not polygon.is_valid or polygon.is_empty:
            return {'error': 'Invalid polygon'}
        
        coords = list(polygon.exterior.coords)
        
        # Основные метрики
        area = polygon.area
        perimeter = polygon.length
        point_count = len(coords)
        
        # Метрики сложности
        if point_count > 2:
            # Коэффициент компактности (4π*площадь/периметр²)
            compactness = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Коэффициент сложности (периметр/√площадь)
            complexity_ratio = perimeter / math.sqrt(area) if area > 0 else 0
            
            # Оценка извилистости границы
            if point_count > 3:
                # Вычисляем среднее отклонение от прямой линии
                deviations = []
                for i in range(1, point_count - 1):
                    p1 = np.array(coords[i-1])
                    p2 = np.array(coords[i])
                    p3 = np.array(coords[i+1])
                    
                    # Расстояние от средней точки до прямой между соседними
                    line_length = np.linalg.norm(p3 - p1)
                    if line_length > 0:
                        # Расстояние от точки до прямой
                        deviation = np.abs(np.cross(p3 - p1, p2 - p1)) / line_length
                        deviations.append(deviation)
                
                avg_deviation = np.mean(deviations) if deviations else 0
                boundary_roughness = avg_deviation / math.sqrt(area) if area > 0 else 0
            else:
                boundary_roughness = 0
        else:
            compactness = 0
            complexity_ratio = 0
            boundary_roughness = 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'point_count': point_count,
            'compactness': compactness,
            'complexity_ratio': complexity_ratio,
            'boundary_roughness': boundary_roughness,
            'is_simple': point_count < 10,
            'is_complex': point_count > 100
        }

def test_adaptive_simplifier():
    """Тестирование адаптивного упростителя"""
    print("🧪 Тестирование адаптивного упростителя полигонов")
    print("=" * 50)
    
    simplifier = AdaptivePolygonSimplifier()
    
    # Создаем тестовые полигоны разной сложности
    test_cases = [
        # Простой полигон (прямоугольник)
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        
        # Сложный полигон (много точек на прямой)
        Polygon([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
                 (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
                 (9, 10), (8, 10), (7, 10), (6, 10), (5, 10), (4, 10), (3, 10), (2, 10), (1, 10), (0, 10),
                 (0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]),
        
        # Очень сложный полигон (симуляция патологического объекта)
        Polygon([(0, 0), (0.5, 0.1), (1, 0.2), (1.5, 0.1), (2, 0), (2.5, 0.1), (3, 0.2), (3.5, 0.1), (4, 0),
                 (4, 0.5), (3.9, 1), (3.8, 1.5), (3.9, 2), (4, 2.5), (4, 3), (3.9, 3.5), (3.8, 4), (3.9, 4.5),
                 (4, 5), (3.5, 5.1), (3, 5.2), (2.5, 5.1), (2, 5), (1.5, 5.1), (1, 5.2), (0.5, 5.1), (0, 5),
                 (0, 4.5), (0.1, 4), (0.2, 3.5), (0.1, 3), (0, 2.5), (0, 2), (0.1, 1.5), (0.2, 1), (0.1, 0.5)])
    ]
    
    for i, polygon in enumerate(test_cases):
        print(f"\n🔍 Тест {i+1}:")
        print(f"   Исходных точек: {len(polygon.exterior.coords)}")
        print(f"   Площадь: {polygon.area:.2f}")
        
        # Анализируем исходный полигон
        original_metrics = simplifier.calculate_polygon_metrics(polygon)
        print(f"   Сложность: {original_metrics['complexity_ratio']:.2f}")
        
        # Упрощаем
        simplified, metrics = simplifier.simplify_polygon(polygon)
        
        print(f"   После упрощения: {metrics['simplified_points']} точек")
        print(f"   Площадь сохранена: {metrics['area_preserved']:.1%}")
        print(f"   Метод: {metrics['method']}")
        
        if metrics['area_preserved'] < 0.95:
            print(f"   ⚠️  Низкое сохранение площади!")
        
        if metrics['simplified_points'] > 100:
            print(f"   ⚠️  Слишком много точек после упрощения!")

if __name__ == "__main__":
    test_adaptive_simplifier()

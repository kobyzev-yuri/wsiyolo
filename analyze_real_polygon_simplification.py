#!/usr/bin/env python3
"""
Анализ алгоритма упрощения полигонов на реальных данных.
Проверяет, правильно ли работает упрощение для патологических объектов (крипт).
"""

import json
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_structures import Coords, Box, Prediction

def load_predictions():
    """Загружает предсказания из JSON файла"""
    try:
        with open('results/predictions.json', 'r') as f:
            data = json.load(f)
        return data['predictions']
    except Exception as e:
        print(f"❌ Ошибка загрузки предсказаний: {e}")
        return []

def analyze_polygon_complexity(polygon_coords):
    """Анализирует сложность полигона"""
    if not polygon_coords or len(polygon_coords) < 3:
        return {
            'point_count': 0,
            'perimeter': 0,
            'area': 0,
            'complexity_ratio': 0,
            'is_simple': False
        }
    
    # Создаем Shapely полигон
    try:
        coords = [(p['x'], p['y']) for p in polygon_coords]
        poly = Polygon(coords)
        
        if not poly.is_valid:
            return {
                'point_count': len(polygon_coords),
                'perimeter': 0,
                'area': 0,
                'complexity_ratio': 0,
                'is_simple': False,
                'error': 'Invalid polygon'
            }
        
        perimeter = poly.length
        area = poly.area
        point_count = len(polygon_coords)
        
        # Коэффициент сложности: периметр/площадь (чем больше, тем сложнее)
        complexity_ratio = perimeter / (area ** 0.5) if area > 0 else 0
        
        return {
            'point_count': point_count,
            'perimeter': perimeter,
            'area': area,
            'complexity_ratio': complexity_ratio,
            'is_simple': poly.is_valid and point_count >= 3,
            'is_valid': poly.is_valid
        }
    except Exception as e:
        return {
            'point_count': len(polygon_coords),
            'perimeter': 0,
            'area': 0,
            'complexity_ratio': 0,
            'is_simple': False,
            'error': str(e)
        }

def smart_simplify_polygon(polygon, max_points=60):
    """Умное упрощение полигона (копия из pipeline)"""
    try:
        current_points = len(polygon.exterior.coords)
        
        if current_points <= max_points:
            return polygon
        
        # Параметры упрощения
        min_tolerance = 0.1
        max_tolerance = 10.0
        best_poly = polygon
        
        # Бинарный поиск оптимального tolerance
        for _ in range(10):
            tolerance = (min_tolerance + max_tolerance) / 2
            simplified = polygon.simplify(tolerance, preserve_topology=True)
            
            if simplified.is_valid and len(simplified.exterior.coords) > 3:
                points_count = len(simplified.exterior.coords)
                
                if points_count <= max_points:
                    best_poly = simplified
                    min_tolerance = tolerance
                    if points_count >= max_points * 0.8:
                        break
                else:
                    max_tolerance = tolerance
            else:
                max_tolerance = tolerance
        
        # Если все еще слишком много точек, используем равномерную выборку
        if len(best_poly.exterior.coords) > max_points:
            coords = list(best_poly.exterior.coords)
            step = len(coords) / max_points
            sampled_coords = []
            
            for i in range(max_points):
                idx = int(i * step) % len(coords)
                sampled_coords.append(coords[idx])
            
            try:
                best_poly = Polygon(sampled_coords)
                if not best_poly.is_valid:
                    best_poly = polygon  # Возвращаем исходный если не удалось
            except:
                best_poly = polygon
        
        return best_poly
        
    except Exception as e:
        print(f"   ⚠️  Ошибка упрощения: {e}")
        return polygon

def analyze_simplification_impact(original_polygon, simplified_polygon):
    """Анализирует влияние упрощения на полигон"""
    try:
        # Метрики исходного полигона
        orig_points = len(original_polygon.exterior.coords)
        orig_perimeter = original_polygon.length
        orig_area = original_polygon.area
        
        # Метрики упрощенного полигона
        simp_points = len(simplified_polygon.exterior.coords)
        simp_perimeter = simplified_polygon.length
        simp_area = simplified_polygon.area
        
        # Вычисляем изменения
        point_reduction = (orig_points - simp_points) / orig_points * 100
        perimeter_change = (orig_perimeter - simp_perimeter) / orig_perimeter * 100
        area_change = (orig_area - simp_area) / orig_area * 100
        
        # Проверяем на патологические случаи
        is_pathological = {
            'too_few_points': simp_points < 6,  # Меньше 6 точек подозрительно
            'excessive_simplification': point_reduction > 90,  # Слишком агрессивное упрощение
            'shape_distortion': abs(area_change) > 20,  # Сильное искажение формы
            'invalid_result': not simplified_polygon.is_valid
        }
        
        return {
            'original': {
                'points': orig_points,
                'perimeter': orig_perimeter,
                'area': orig_area
            },
            'simplified': {
                'points': simp_points,
                'perimeter': simp_perimeter,
                'area': simp_area
            },
            'changes': {
                'point_reduction_pct': point_reduction,
                'perimeter_change_pct': perimeter_change,
                'area_change_pct': area_change
            },
            'pathological': is_pathological,
            'is_problematic': any(is_pathological.values())
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'is_problematic': True
        }

def main():
    """Основная функция анализа"""
    print("🔍 Анализ алгоритма упрощения полигонов на реальных данных")
    print("=" * 60)
    
    # Загружаем предсказания
    predictions = load_predictions()
    if not predictions:
        print("❌ Не удалось загрузить предсказания")
        return
    
    print(f"📊 Загружено {len(predictions)} предсказаний")
    
    # Анализируем полигоны
    analysis_results = []
    problematic_cases = []
    
    for i, pred in enumerate(predictions[:100]):  # Анализируем первые 100
        if 'polygon' not in pred or not pred['polygon']:
            continue
            
        polygon_coords = pred['polygon']
        class_name = pred.get('class_name', 'unknown')
        
        # Анализируем исходную сложность
        complexity = analyze_polygon_complexity(polygon_coords)
        
        if not complexity['is_simple']:
            continue
        
        # Создаем Shapely полигон
        try:
            coords = [(p['x'], p['y']) for p in polygon_coords]
            original_polygon = Polygon(coords)
            
            if not original_polygon.is_valid:
                continue
            
            # Упрощаем полигон
            simplified_polygon = smart_simplify_polygon(original_polygon)
            
            # Анализируем влияние упрощения
            impact = analyze_simplification_impact(original_polygon, simplified_polygon)
            
            result = {
                'index': i,
                'class': class_name,
                'complexity': complexity,
                'impact': impact
            }
            
            analysis_results.append(result)
            
            # Проверяем на проблемные случаи
            if impact.get('is_problematic', False):
                problematic_cases.append(result)
                print(f"⚠️  Проблемный случай {i} ({class_name}):")
                print(f"   Исходных точек: {impact['original']['points']}")
                print(f"   После упрощения: {impact['simplified']['points']}")
                print(f"   Сокращение: {impact['changes']['point_reduction_pct']:.1f}%")
                print(f"   Проблемы: {[k for k, v in impact['pathological'].items() if v]}")
                print()
                
        except Exception as e:
            print(f"❌ Ошибка обработки предсказания {i}: {e}")
            continue
    
    # Статистика
    print(f"\n📈 Статистика анализа:")
    print(f"   Проанализировано полигонов: {len(analysis_results)}")
    print(f"   Проблемных случаев: {len(problematic_cases)}")
    
    if analysis_results:
        point_reductions = [r['impact']['changes']['point_reduction_pct'] for r in analysis_results]
        area_changes = [abs(r['impact']['changes']['area_change_pct']) for r in analysis_results]
        
        print(f"   Среднее сокращение точек: {np.mean(point_reductions):.1f}%")
        print(f"   Максимальное сокращение: {np.max(point_reductions):.1f}%")
        print(f"   Среднее изменение площади: {np.mean(area_changes):.1f}%")
        print(f"   Максимальное изменение площади: {np.max(area_changes):.1f}%")
    
    # Анализ по классам
    class_stats = {}
    for result in analysis_results:
        class_name = result['class']
        if class_name not in class_stats:
            class_stats[class_name] = {'total': 0, 'problematic': 0}
        
        class_stats[class_name]['total'] += 1
        if result['impact'].get('is_problematic', False):
            class_stats[class_name]['problematic'] += 1
    
    print(f"\n📊 Статистика по классам:")
    for class_name, stats in class_stats.items():
        problematic_pct = stats['problematic'] / stats['total'] * 100
        print(f"   {class_name}: {stats['total']} полигонов, {problematic_pct:.1f}% проблемных")
    
    # Рекомендации
    print(f"\n💡 Рекомендации:")
    if len(problematic_cases) > len(analysis_results) * 0.1:  # Более 10% проблемных
        print("   ⚠️  Высокий процент проблемных случаев - требуется доработка алгоритма")
        print("   🔧 Рекомендуется:")
        print("      - Увеличить min_tolerance для менее агрессивного упрощения")
        print("      - Добавить проверку на минимальное количество точек")
        print("      - Использовать адаптивные параметры в зависимости от сложности полигона")
    else:
        print("   ✅ Алгоритм работает корректно")
    
    return analysis_results, problematic_cases

if __name__ == "__main__":
    main()

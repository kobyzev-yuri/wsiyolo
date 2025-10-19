#!/usr/bin/env python3
"""
Анализ улучшенного алгоритма упрощения полигонов на реальных данных.
Использует адаптивный подход с фокусом на сохранение точности площади.
"""

import json
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adaptive_polygon_simplifier import AdaptivePolygonSimplifier

def load_predictions():
    """Загружает предсказания из JSON файла"""
    try:
        with open('results/predictions.json', 'r') as f:
            data = json.load(f)
        return data['predictions']
    except Exception as e:
        print(f"❌ Ошибка загрузки предсказаний: {e}")
        return []

def analyze_polygon_with_improved_algorithm(polygon_coords, class_name="unknown"):
    """Анализирует полигон с улучшенным алгоритмом"""
    if not polygon_coords or len(polygon_coords) < 3:
        return None
    
    try:
        # Создаем Shapely полигон
        coords = [(p['x'], p['y']) for p in polygon_coords]
        original_polygon = Polygon(coords)
        
        if not original_polygon.is_valid:
            return None
        
        # Используем улучшенный алгоритм
        simplifier = AdaptivePolygonSimplifier()
        simplified_polygon, metrics = simplifier.simplify_polygon(original_polygon)
        
        # Анализируем результат
        original_metrics = simplifier.calculate_polygon_metrics(original_polygon)
        simplified_metrics = simplifier.calculate_polygon_metrics(simplified_polygon)
        
        # Проверяем на проблемы
        problems = []
        
        # Проверка 1: Слишком агрессивное упрощение
        if metrics['area_preserved'] < 0.95:
            problems.append('area_loss')
        
        # Проверка 2: Слишком много точек после упрощения
        if metrics['simplified_points'] > 100:
            problems.append('too_many_points')
        
        # Проверка 3: Слишком мало исходных точек (подозрительно)
        if metrics['original_points'] < 6:
            problems.append('too_few_original_points')
        
        # Проверка 4: Сильное искажение формы
        if abs(metrics['area_preserved'] - 1.0) > 0.1:
            problems.append('shape_distortion')
        
        return {
            'class_name': class_name,
            'original_points': metrics['original_points'],
            'simplified_points': metrics['simplified_points'],
            'area_preserved': metrics['area_preserved'],
            'perimeter_preserved': metrics['perimeter_preserved'],
            'method': metrics['method'],
            'complexity': metrics['complexity'],
            'problems': problems,
            'is_problematic': len(problems) > 0,
            'original_metrics': original_metrics,
            'simplified_metrics': simplified_metrics
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'is_problematic': True
        }

def compare_algorithms(polygon_coords, class_name="unknown"):
    """Сравнивает старый и новый алгоритмы"""
    if not polygon_coords or len(polygon_coords) < 3:
        return None
    
    try:
        coords = [(p['x'], p['y']) for p in polygon_coords]
        original_polygon = Polygon(coords)
        
        if not original_polygon.is_valid:
            return None
        
        # Старый алгоритм (из pipeline)
        old_simplified = old_simplify_polygon(original_polygon)
        old_metrics = {
            'points': len(old_simplified.exterior.coords),
            'area_preserved': old_simplified.area / original_polygon.area,
            'perimeter_preserved': old_simplified.length / original_polygon.length
        }
        
        # Новый алгоритм
        simplifier = AdaptivePolygonSimplifier()
        new_simplified, new_metrics = simplifier.simplify_polygon(original_polygon)
        
        return {
            'class_name': class_name,
            'original_points': len(original_polygon.exterior.coords),
            'old_algorithm': old_metrics,
            'new_algorithm': {
                'points': new_metrics['simplified_points'],
                'area_preserved': new_metrics['area_preserved'],
                'perimeter_preserved': new_metrics['perimeter_preserved'],
                'method': new_metrics['method']
            },
            'improvement': {
                'area_preservation_gain': new_metrics['area_preserved'] - old_metrics['area_preserved'],
                'points_reduction_old': (len(original_polygon.exterior.coords) - old_metrics['points']) / len(original_polygon.exterior.coords),
                'points_reduction_new': (len(original_polygon.exterior.coords) - new_metrics['simplified_points']) / len(original_polygon.exterior.coords)
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

def old_simplify_polygon(polygon, max_points=60):
    """Старый алгоритм упрощения (копия из pipeline)"""
    try:
        current_points = len(polygon.exterior.coords)
        
        if current_points <= max_points:
            return polygon
        
        min_tolerance = 0.1
        max_tolerance = 10.0
        best_poly = polygon
        
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
        
        # Равномерная выборка если нужно
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
                    best_poly = polygon
            except:
                best_poly = polygon
        
        return best_poly
        
    except Exception as e:
        return polygon

def main():
    """Основная функция анализа"""
    print("🔍 Анализ улучшенного алгоритма упрощения полигонов")
    print("=" * 60)
    
    # Загружаем предсказания
    predictions = load_predictions()
    if not predictions:
        print("❌ Не удалось загрузить предсказания")
        return
    
    print(f"📊 Загружено {len(predictions)} предсказаний")
    
    # Анализируем с новым алгоритмом
    print("\n🔧 Анализ с улучшенным алгоритмом:")
    new_results = []
    new_problematic = []
    
    for i, pred in enumerate(predictions[:100]):  # Первые 100
        if 'polygon' not in pred or not pred['polygon']:
            continue
            
        class_name = pred.get('class_name', 'unknown')
        result = analyze_polygon_with_improved_algorithm(pred['polygon'], class_name)
        
        if result and not result.get('error'):
            new_results.append(result)
            if result.get('is_problematic', False):
                new_problematic.append(result)
                print(f"⚠️  Проблемный случай {i} ({class_name}):")
                print(f"   Исходных точек: {result['original_points']}")
                print(f"   После упрощения: {result['simplified_points']}")
                print(f"   Площадь сохранена: {result['area_preserved']:.1%}")
                print(f"   Проблемы: {result['problems']}")
                print()
    
    # Сравниваем алгоритмы
    print("\n🔄 Сравнение алгоритмов:")
    comparison_results = []
    
    for i, pred in enumerate(predictions[:50]):  # Первые 50 для сравнения
        if 'polygon' not in pred or not pred['polygon']:
            continue
            
        class_name = pred.get('class_name', 'unknown')
        comparison = compare_algorithms(pred['polygon'], class_name)
        
        if comparison and not comparison.get('error'):
            comparison_results.append(comparison)
    
    # Статистика нового алгоритма
    print(f"\n📈 Статистика улучшенного алгоритма:")
    print(f"   Проанализировано: {len(new_results)}")
    print(f"   Проблемных случаев: {len(new_problematic)}")
    
    if new_results:
        area_preserved = [r['area_preserved'] for r in new_results]
        points_reduction = [(r['original_points'] - r['simplified_points']) / r['original_points'] * 100 for r in new_results]
        
        print(f"   Среднее сохранение площади: {np.mean(area_preserved):.1%}")
        print(f"   Минимальное сохранение площади: {np.min(area_preserved):.1%}")
        print(f"   Среднее сокращение точек: {np.mean(points_reduction):.1f}%")
        print(f"   Максимальное сокращение: {np.max(points_reduction):.1f}%")
    
    # Сравнение с старым алгоритмом
    if comparison_results:
        print(f"\n📊 Сравнение с старым алгоритмом:")
        area_gains = [c['improvement']['area_preservation_gain'] for c in comparison_results]
        print(f"   Средний прирост сохранения площади: {np.mean(area_gains):.1%}")
        print(f"   Случаев с улучшением: {sum(1 for g in area_gains if g > 0)}/{len(area_gains)}")
    
    # Анализ по классам
    class_stats = {}
    for result in new_results:
        class_name = result['class_name']
        if class_name not in class_stats:
            class_stats[class_name] = {'total': 0, 'problematic': 0, 'avg_area_preserved': []}
        
        class_stats[class_name]['total'] += 1
        class_stats[class_name]['avg_area_preserved'].append(result['area_preserved'])
        if result.get('is_problematic', False):
            class_stats[class_name]['problematic'] += 1
    
    print(f"\n📊 Статистика по классам:")
    for class_name, stats in class_stats.items():
        problematic_pct = stats['problematic'] / stats['total'] * 100
        avg_area = np.mean(stats['avg_area_preserved'])
        print(f"   {class_name}: {stats['total']} полигонов, {problematic_pct:.1f}% проблемных, {avg_area:.1%} сохранение площади")
    
    # Рекомендации
    print(f"\n💡 Рекомендации:")
    if len(new_problematic) > len(new_results) * 0.05:  # Более 5% проблемных
        print("   ⚠️  Есть проблемные случаи - требуется дополнительная настройка")
    else:
        print("   ✅ Улучшенный алгоритм работает значительно лучше!")
        print("   🎯 Рекомендуется интегрировать в основной pipeline")

if __name__ == "__main__":
    main()

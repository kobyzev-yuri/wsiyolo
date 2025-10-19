#!/usr/bin/env python3
"""
Сравнение старого и нового WSI YOLO Pipeline.
Тестирует производительность, качество и точность результатов.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Импорты для старого pipeline
from wsi_yolo_pipeline import WSIYOLOPipeline

# Импорты для нового pipeline
from improved_wsi_yolo_pipeline import ImprovedWSIYOLOPipeline

def load_predictions(file_path: str) -> Dict[str, Any]:
    """Загружает предсказания из JSON файла"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки {file_path}: {e}")
        return {}

def compare_predictions(old_predictions: List[Dict], new_predictions: List[Dict]) -> Dict[str, Any]:
    """Сравнивает предсказания двух pipeline"""
    
    # Статистика по количеству
    old_count = len(old_predictions)
    new_count = len(new_predictions)
    
    # Статистика по классам
    old_classes = {}
    new_classes = {}
    
    for pred in old_predictions:
        class_name = pred.get('class_name', 'unknown')
        old_classes[class_name] = old_classes.get(class_name, 0) + 1
    
    for pred in new_predictions:
        class_name = pred.get('class_name', 'unknown')
        new_classes[class_name] = new_classes.get(class_name, 0) + 1
    
    # Анализ полигонов
    old_polygon_stats = analyze_polygon_statistics(old_predictions)
    new_polygon_stats = analyze_polygon_statistics(new_predictions)
    
    return {
        'count_comparison': {
            'old_count': old_count,
            'new_count': new_count,
            'difference': new_count - old_count,
            'change_percent': (new_count - old_count) / old_count * 100 if old_count > 0 else 0
        },
        'class_comparison': {
            'old_classes': old_classes,
            'new_classes': new_classes
        },
        'polygon_analysis': {
            'old_stats': old_polygon_stats,
            'new_stats': new_polygon_stats
        }
    }

def analyze_polygon_statistics(predictions: List[Dict]) -> Dict[str, Any]:
    """Анализирует статистику полигонов"""
    if not predictions:
        return {}
    
    polygon_counts = []
    areas = []
    perimeters = []
    simplification_metrics = []
    
    for pred in predictions:
        if pred.get('polygon'):
            polygon_counts.append(len(pred['polygon']))
            
            # Вычисляем площадь и периметр (упрощенно)
            if len(pred['polygon']) >= 3:
                # Простое вычисление площади через shoelace formula
                coords = [(p['x'], p['y']) for p in pred['polygon']]
                area = calculate_polygon_area(coords)
                perimeter = calculate_polygon_perimeter(coords)
                
                areas.append(area)
                perimeters.append(perimeter)
        
        # Анализируем метрики упрощения если есть
        if 'simplification_metrics' in pred:
            metrics = pred['simplification_metrics']
            simplification_metrics.append(metrics)
    
    stats = {
        'total_predictions': len(predictions),
        'predictions_with_polygons': len([p for p in predictions if p.get('polygon')]),
        'avg_polygon_points': np.mean(polygon_counts) if polygon_counts else 0,
        'max_polygon_points': max(polygon_counts) if polygon_counts else 0,
        'min_polygon_points': min(polygon_counts) if polygon_counts else 0
    }
    
    if areas:
        stats.update({
            'avg_area': np.mean(areas),
            'total_area': np.sum(areas),
            'avg_perimeter': np.mean(perimeters)
        })
    
    if simplification_metrics:
        area_preserved = [m.get('area_preserved', 1.0) for m in simplification_metrics]
        stats.update({
            'avg_area_preserved': np.mean(area_preserved),
            'min_area_preserved': np.min(area_preserved),
            'simplification_applied': len(simplification_metrics)
        })
    
    return stats

def calculate_polygon_area(coords: List[tuple]) -> float:
    """Вычисляет площадь полигона по формуле шнурка"""
    if len(coords) < 3:
        return 0.0
    
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def calculate_polygon_perimeter(coords: List[tuple]) -> float:
    """Вычисляет периметр полигона"""
    if len(coords) < 2:
        return 0.0
    
    perimeter = 0.0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        dx = coords[j][0] - coords[i][0]
        dy = coords[j][1] - coords[i][1]
        perimeter += np.sqrt(dx*dx + dy*dy)
    
    return perimeter

def compare_performance(old_stats: Dict, new_stats: Dict) -> Dict[str, Any]:
    """Сравнивает производительность pipeline"""
    
    old_time = old_stats.get('processing_time', 0)
    new_time = new_stats.get('processing_time', 0)
    
    old_patches = old_stats.get('total_patches', 0)
    new_patches = new_stats.get('total_patches', 0)
    
    return {
        'time_comparison': {
            'old_time': old_time,
            'new_time': new_time,
            'speedup': old_time / new_time if new_time > 0 else 0,
            'time_saved': old_time - new_time
        },
        'throughput_comparison': {
            'old_throughput': old_patches / old_time if old_time > 0 else 0,
            'new_throughput': new_patches / new_time if new_time > 0 else 0,
            'throughput_improvement': (new_patches / new_time) / (old_patches / old_time) if old_time > 0 and new_time > 0 else 0
        }
    }

def main():
    """Основная функция сравнения"""
    print("🔄 Сравнение старого и нового WSI YOLO Pipeline")
    print("=" * 60)
    
    # Пути к результатам
    old_results_path = "results/predictions.json"
    new_results_path = "results_improved_full/improved_predictions.json"
    
    # Загружаем результаты
    print("📊 Загрузка результатов...")
    old_results = load_predictions(old_results_path)
    new_results = load_predictions(new_results_path)
    
    if not old_results or not new_results:
        print("❌ Не удалось загрузить результаты для сравнения")
        return
    
    old_predictions = old_results.get('predictions', [])
    new_predictions = new_results.get('predictions', [])
    
    print(f"   Старый pipeline: {len(old_predictions)} предсказаний")
    print(f"   Новый pipeline: {len(new_predictions)} предсказаний")
    
    # Сравниваем предсказания
    print("\n🔍 Анализ предсказаний...")
    prediction_comparison = compare_predictions(old_predictions, new_predictions)
    
    # Выводим результаты
    print(f"\n📈 Сравнение количества предсказаний:")
    count_comp = prediction_comparison['count_comparison']
    print(f"   Старый: {count_comp['old_count']}")
    print(f"   Новый: {count_comp['new_count']}")
    print(f"   Разница: {count_comp['difference']} ({count_comp['change_percent']:+.1f}%)")
    
    # Сравнение по классам
    print(f"\n📊 Сравнение по классам:")
    class_comp = prediction_comparison['class_comparison']
    all_classes = set(class_comp['old_classes'].keys()) | set(class_comp['new_classes'].keys())
    
    for class_name in sorted(all_classes):
        old_count = class_comp['old_classes'].get(class_name, 0)
        new_count = class_comp['new_classes'].get(class_name, 0)
        change = new_count - old_count
        change_pct = (change / old_count * 100) if old_count > 0 else 0
        print(f"   {class_name}: {old_count} → {new_count} ({change:+.0f}, {change_pct:+.1f}%)")
    
    # Анализ полигонов
    print(f"\n🔍 Анализ полигонов:")
    old_poly_stats = prediction_comparison['polygon_analysis']['old_stats']
    new_poly_stats = prediction_comparison['polygon_analysis']['new_stats']
    
    print(f"   Среднее количество точек в полигоне:")
    print(f"     Старый: {old_poly_stats.get('avg_polygon_points', 0):.1f}")
    print(f"     Новый: {new_poly_stats.get('avg_polygon_points', 0):.1f}")
    
    if 'avg_area_preserved' in new_poly_stats:
        print(f"   Сохранение площади (новый): {new_poly_stats['avg_area_preserved']:.1%}")
        print(f"   Минимальное сохранение: {new_poly_stats.get('min_area_preserved', 0):.1%}")
    
    # Сравнение производительности
    print(f"\n⚡ Сравнение производительности:")
    old_perf = old_results.get('performance_stats', {})
    new_perf = new_results.get('performance_stats', {})
    
    if old_perf and new_perf:
        perf_comparison = compare_performance(old_perf, new_perf)
        
        time_comp = perf_comparison['time_comparison']
        print(f"   Время обработки:")
        print(f"     Старый: {time_comp['old_time']:.2f}с")
        print(f"     Новый: {time_comp['new_time']:.2f}с")
        print(f"     Ускорение: {time_comp['speedup']:.2f}x")
        print(f"     Экономия времени: {time_comp['time_saved']:.2f}с")
        
        throughput_comp = perf_comparison['throughput_comparison']
        print(f"   Пропускная способность:")
        print(f"     Старый: {throughput_comp['old_throughput']:.1f} патчей/сек")
        print(f"     Новый: {throughput_comp['new_throughput']:.1f} патчей/сек")
        print(f"     Улучшение: {throughput_comp['throughput_improvement']:.2f}x")
    
    # Рекомендации
    print(f"\n💡 Рекомендации:")
    if count_comp['change_percent'] > 10:
        print("   ⚠️  Значительное изменение количества предсказаний - проверьте фильтрацию")
    elif count_comp['change_percent'] < -10:
        print("   ⚠️  Значительное уменьшение предсказаний - проверьте алгоритм объединения")
    else:
        print("   ✅ Количество предсказаний стабильно")
    
    if 'avg_area_preserved' in new_poly_stats and new_poly_stats['avg_area_preserved'] > 0.95:
        print("   ✅ Отличное сохранение площади полигонов")
    elif 'avg_area_preserved' in new_poly_stats:
        print("   ⚠️  Снижение сохранения площади - требуется настройка")
    
    if old_perf and new_perf:
        speedup = perf_comparison['time_comparison']['speedup']
        if speedup > 1.5:
            print(f"   ✅ Значительное ускорение: {speedup:.1f}x")
        elif speedup > 1.1:
            print(f"   ✅ Умеренное ускорение: {speedup:.1f}x")
        else:
            print("   ⚠️  Незначительное ускорение - требуется оптимизация")

if __name__ == "__main__":
    main()

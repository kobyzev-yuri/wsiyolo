#!/usr/bin/env python3
"""
Скрипт для анализа проблемы с огромным полигоном класса excl
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

def analyze_polygon_issue(json_file_path):
    """Анализирует проблему с полигоном в JSON файле"""
    
    print(f"🔍 Анализ файла: {json_file_path}")
    
    # Загружаем JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Статистика:")
    print(f"   Всего предсказаний: {data['statistics']['total']}")
    print(f"   Классы: {data['statistics']['by_class']}")
    print(f"   Средняя уверенность: {data['statistics']['average_confidence']}")
    
    # Анализируем предсказания
    predictions = data['predictions']
    
    for i, pred in enumerate(predictions):
        print(f"\n🎯 Предсказание {i+1}:")
        print(f"   Класс: {pred['class_name']}")
        print(f"   Уверенность: {pred['confidence']}")
        
        # Анализируем bbox
        box = pred['box']
        width = box['end']['x'] - box['start']['x']
        height = box['end']['y'] - box['start']['y']
        print(f"   Bbox: {width:.1f} x {height:.1f} (площадь: {width*height:.1f})")
        
        # Анализируем полигон
        polygon = pred['polygon']
        print(f"   Количество точек полигона: {len(polygon)}")
        
        if len(polygon) > 1000:
            print(f"   ⚠️  ОГРОМНЫЙ ПОЛИГОН! {len(polygon)} точек")
            
            # Анализируем координаты
            x_coords = [p['x'] for p in polygon]
            y_coords = [p['y'] for p in polygon]
            
            print(f"   X диапазон: {min(x_coords):.1f} - {max(x_coords):.1f}")
            print(f"   Y диапазон: {min(y_coords):.1f} - {max(y_coords):.1f}")
            
            # Проверяем на дубликаты
            unique_points = set((x, y) for x, y in zip(x_coords, y_coords))
            print(f"   Уникальных точек: {len(unique_points)} из {len(polygon)}")
            
            # Ищем паттерны в координатах
            print(f"   Первые 10 точек:")
            for j in range(min(10, len(polygon))):
                print(f"     {j}: ({polygon[j]['x']:.1f}, {polygon[j]['y']:.1f})")
            
            # Проверяем на повторяющиеся паттерны
            if len(polygon) > 100:
                # Ищем повторяющиеся последовательности
                pattern_found = False
                for pattern_len in [10, 20, 50, 100]:
                    if len(polygon) > pattern_len * 2:
                        first_pattern = polygon[:pattern_len]
                        second_pattern = polygon[pattern_len:pattern_len*2]
                        if first_pattern == second_pattern:
                            print(f"   🔄 Найден повторяющийся паттерн длиной {pattern_len}")
                            pattern_found = True
                            break
                
                if not pattern_found:
                    print(f"   ❓ Нет очевидных повторяющихся паттернов")
            
            # Анализируем геометрию
            try:
                coords = [(p['x'], p['y']) for p in polygon]
                shapely_poly = Polygon(coords)
                
                if shapely_poly.is_valid:
                    print(f"   ✅ Полигон валиден")
                    print(f"   Площадь: {shapely_poly.area:.1f}")
                    print(f"   Периметр: {shapely_poly.length:.1f}")
                else:
                    print(f"   ❌ Полигон невалиден")
                    print(f"   Причина: {shapely_poly.validity_reason}")
                    
            except Exception as e:
                print(f"   ❌ Ошибка создания Shapely полигона: {e}")
    
    return data

def check_polygon_merger_logic():
    """Проверяет логику PolygonMerger"""
    
    print(f"\n🔧 Анализ логики PolygonMerger:")
    
    # Создаем тестовые полигоны
    test_polygons = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(5, 5), (15, 5), (15, 15), (5, 15)]),
        Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    ]
    
    print(f"   Тестовые полигоны: {len(test_polygons)}")
    for i, poly in enumerate(test_polygons):
        print(f"     Полигон {i+1}: площадь={poly.area:.1f}, периметр={poly.length:.1f}")
    
    # Тестируем unary_union
    try:
        merged = unary_union(test_polygons)
        print(f"   ✅ unary_union работает")
        print(f"   Результат: {type(merged)}")
        if hasattr(merged, 'area'):
            print(f"   Площадь объединенного: {merged.area:.1f}")
        if hasattr(merged, 'length'):
            print(f"   Периметр объединенного: {merged.length:.1f}")
            
    except Exception as e:
        print(f"   ❌ Ошибка unary_union: {e}")

def create_visualization(json_file_path):
    """Создает визуализацию полигона"""
    
    print(f"\n📊 Создание визуализации...")
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    
    for i, pred in enumerate(predictions):
        if len(pred['polygon']) > 1000:
            print(f"   Создаем визуализацию для предсказания {i+1}")
            
            # Извлекаем координаты
            x_coords = [p['x'] for p in pred['polygon']]
            y_coords = [p['y'] for p in pred['polygon']]
            
            # Создаем график
            plt.figure(figsize=(12, 8))
            plt.plot(x_coords, y_coords, 'b-', linewidth=0.5, alpha=0.7)
            plt.scatter(x_coords[::100], y_coords[::100], c='red', s=1, alpha=0.5)
            plt.title(f'Полигон класса {pred["class_name"]} ({len(pred["polygon"])} точек)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # Сохраняем
            output_path = f"polygon_visualization_{i+1}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   Сохранено: {output_path}")
            plt.close()

if __name__ == "__main__":
    json_file = "/mnt/ai/cnn/wsiyolo/results/19_ibd_mod_S037__20240822_091343.json"
    
    # Анализируем проблему
    data = analyze_polygon_issue(json_file)
    
    # Проверяем логику мержера
    check_polygon_merger_logic()
    
    # Создаем визуализацию
    create_visualization(json_file)
    
    print(f"\n✅ Анализ завершен")


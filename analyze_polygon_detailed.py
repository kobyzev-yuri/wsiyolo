#!/usr/bin/env python3
"""
Детальный анализ проблемы с полигоном
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage import measure

def analyze_mask_to_polygon_issue():
    """Анализирует проблему в функции _mask_to_polygon"""
    
    print("🔍 Анализ функции _mask_to_polygon:")
    
    # Создаем тестовую маску с проблемой
    # Симулируем маску, которая может дать много точек
    mask = np.zeros((512, 512), dtype=np.float32)
    
    # Создаем сложную форму с множеством деталей
    # Это может произойти, если YOLO создает очень детализированную маску
    for i in range(100, 400):
        for j in range(100, 400):
            # Создаем "шумную" маску с множеством мелких деталей
            if (i - 250)**2 + (j - 250)**2 < 10000:  # Круг
                mask[i, j] = 1.0
                # Добавляем шум
                if (i + j) % 3 == 0:
                    mask[i, j] = 0.8
                if (i * j) % 5 == 0:
                    mask[i, j] = 0.9
    
    print(f"   Размер маски: {mask.shape}")
    print(f"   Уникальных значений: {len(np.unique(mask))}")
    print(f"   Минимум: {mask.min()}, Максимум: {mask.max()}")
    
    # Тестируем find_contours
    try:
        contours = measure.find_contours(mask, 0.5)
        print(f"   Найдено контуров: {len(contours)}")
        
        if contours:
            largest_contour = max(contours, key=len)
            print(f"   Размер самого большого контура: {len(largest_contour)}")
            
            # Анализируем контур
            print(f"   Первые 10 точек контура:")
            for i in range(min(10, len(largest_contour))):
                print(f"     {i}: ({largest_contour[i][1]:.1f}, {largest_contour[i][0]:.1f})")
            
            # Проверяем на дубликаты
            unique_points = set((p[1], p[0]) for p in largest_contour)
            print(f"   Уникальных точек в контуре: {len(unique_points)} из {len(largest_contour)}")
            
            # Создаем полигон
            coords = [(p[1], p[0]) for p in largest_contour]
            poly = Polygon(coords)
            
            if poly.is_valid:
                print(f"   ✅ Полигон валиден")
                print(f"   Площадь: {poly.area:.1f}")
                print(f"   Периметр: {poly.length:.1f}")
            else:
                print(f"   ❌ Полигон невалиден: {poly.validity_reason}")
                
    except Exception as e:
        print(f"   ❌ Ошибка анализа контуров: {e}")

def analyze_actual_polygon_structure():
    """Анализирует структуру реального полигона из JSON"""
    
    print("\n🔍 Анализ структуры реального полигона:")
    
    json_file = "/mnt/ai/cnn/wsiyolo/results/19_ibd_mod_S037__20240822_091343.json"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pred = data['predictions'][0]
    polygon = pred['polygon']
    
    print(f"   Всего точек: {len(polygon)}")
    
    # Анализируем паттерны в координатах
    x_coords = [p['x'] for p in polygon]
    y_coords = [p['y'] for p in polygon]
    
    print(f"   X координаты:")
    print(f"     Минимум: {min(x_coords):.1f}")
    print(f"     Максимум: {max(x_coords):.1f}")
    print(f"     Диапазон: {max(x_coords) - min(x_coords):.1f}")
    
    print(f"   Y координаты:")
    print(f"     Минимум: {min(y_coords):.1f}")
    print(f"     Максимум: {max(y_coords):.1f}")
    print(f"     Диапазон: {max(y_coords) - min(y_coords):.1f}")
    
    # Ищем паттерны
    print(f"\n   Анализ паттернов:")
    
    # Проверяем на последовательные значения
    x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
    y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
    
    unique_x_diffs = set(x_diffs)
    unique_y_diffs = set(y_diffs)
    
    print(f"   Уникальных X разностей: {len(unique_x_diffs)}")
    print(f"   Уникальных Y разностей: {len(unique_y_diffs)}")
    
    # Ищем повторяющиеся разности
    from collections import Counter
    x_diff_counts = Counter(x_diffs)
    y_diff_counts = Counter(y_diffs)
    
    print(f"   Наиболее частые X разности:")
    for diff, count in x_diff_counts.most_common(5):
        print(f"     {diff:.1f}: {count} раз")
    
    print(f"   Наиболее частые Y разности:")
    for diff, count in y_diff_counts.most_common(5):
        print(f"     {diff:.1f}: {count} раз")
    
    # Проверяем на "пиксельные" паттерны
    pixel_like_x = all(diff in [-1, 0, 1] for diff in unique_x_diffs)
    pixel_like_y = all(diff in [-1, 0, 1] for diff in unique_y_diffs)
    
    print(f"   Пиксельные паттерны:")
    print(f"     X: {'✅' if pixel_like_x else '❌'}")
    print(f"     Y: {'✅' if pixel_like_y else '❌'}")

def test_polygon_simplification():
    """Тестирует упрощение полигона"""
    
    print("\n🔧 Тестирование упрощения полигона:")
    
    # Создаем сложный полигон
    coords = []
    for i in range(1000):
        angle = i * 0.1
        x = 100 + 50 * np.cos(angle) + np.random.normal(0, 0.1)
        y = 100 + 50 * np.sin(angle) + np.random.normal(0, 0.1)
        coords.append((x, y))
    
    poly = Polygon(coords)
    print(f"   Исходный полигон: {len(coords)} точек, периметр: {poly.length:.1f}")
    
    # Упрощаем полигон
    try:
        simplified = poly.simplify(1.0, preserve_topology=True)
        print(f"   Упрощенный полигон: {len(simplified.exterior.coords)-1} точек, периметр: {simplified.length:.1f}")
        print(f"   Коэффициент упрощения: {len(simplified.exterior.coords)/len(coords):.3f}")
        
    except Exception as e:
        print(f"   ❌ Ошибка упрощения: {e}")

def suggest_fixes():
    """Предлагает исправления"""
    
    print("\n💡 Предлагаемые исправления:")
    
    print("1. Упрощение полигонов в _mask_to_polygon:")
    print("   - Добавить simplify() после создания полигона")
    print("   - Ограничить количество точек")
    print("   - Использовать Douglas-Peucker алгоритм")
    
    print("\n2. Улучшение в PolygonMerger:")
    print("   - Добавить упрощение перед unary_union")
    print("   - Ограничить количество точек в результирующем полигоне")
    print("   - Добавить валидацию размера полигона")
    
    print("\n3. Фильтрация в YOLO inference:")
    print("   - Добавить минимальную площадь маски")
    print("   - Фильтровать слишком детализированные маски")
    print("   - Использовать морфологические операции")

if __name__ == "__main__":
    analyze_mask_to_polygon_issue()
    analyze_actual_polygon_structure()
    test_polygon_simplification()
    suggest_fixes()
    
    print("\n✅ Детальный анализ завершен")


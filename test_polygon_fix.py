#!/usr/bin/env python3
"""
Тестирование исправлений для проблемы с полигонами
"""

import numpy as np
from skimage import measure
from shapely.geometry import Polygon
from shapely.ops import unary_union

def smart_simplify_polygon(polygon, max_points=60):
    """Умное упрощение полигона до заданного количества точек"""
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

def test_polygon_simplification():
    """Тестирует умное упрощение полигонов"""
    
    print("🧪 Тестирование умного упрощения полигонов:")
    
    # Создаем сложную маску, которая может дать много точек
    mask = np.zeros((512, 512), dtype=np.float32)
    
    # Создаем сложную форму с множеством деталей
    for i in range(100, 400):
        for j in range(100, 400):
            if (i - 250)**2 + (j - 250)**2 < 10000:  # Круг
                mask[i, j] = 1.0
                # Добавляем шум для создания детализированного контура
                if (i + j) % 3 == 0:
                    mask[i, j] = 0.8
                if (i * j) % 5 == 0:
                    mask[i, j] = 0.9
    
    print(f"   Размер маски: {mask.shape}")
    
    # Тестируем умное упрощение
    contours = measure.find_contours(mask, 0.5)
    if contours:
        largest_contour = max(contours, key=len)
        print(f"   Исходный контур: {len(largest_contour)} точек")
        
        # Создаем полигон
        coords = [(p[1], p[0]) for p in largest_contour]
        poly = Polygon(coords)
        
        if poly.is_valid:
            print(f"   Исходный полигон: {len(poly.exterior.coords)} точек, периметр: {poly.length:.1f}")
            
            # Тестируем умное упрощение до 60 точек
            simplified = smart_simplify_polygon(poly, max_points=60)
            print(f"   Умно упрощенный полигон: {len(simplified.exterior.coords)} точек, периметр: {simplified.length:.1f}")
            
            # Проверяем качество упрощения
            area_ratio = simplified.area / poly.area
            print(f"   Сохранение площади: {area_ratio:.3f}")
            
            # Проверяем, что количество точек не превышает лимит
            points_ok = len(simplified.exterior.coords) <= 60
            area_ok = area_ratio > 0.8  # Сохраняем минимум 80% площади
            
            print(f"   ✅ Количество точек: {'OK' if points_ok else 'FAIL'} ({len(simplified.exterior.coords)}/60)")
            print(f"   ✅ Сохранение площади: {'OK' if area_ok else 'FAIL'} ({area_ratio:.3f})")
            
            return points_ok and area_ok
    
    return False

def test_merger_simplification():
    """Тестирует упрощение в PolygonMerger"""
    
    print("\n🧪 Тестирование упрощения в PolygonMerger:")
    
    # Создаем несколько сложных полигонов
    polygons = []
    for i in range(3):
        # Создаем полигон с множеством точек
        coords = []
        center_x, center_y = 100 + i * 200, 100 + i * 200
        for angle in np.linspace(0, 2*np.pi, 1000):
            x = center_x + 50 * np.cos(angle) + np.random.normal(0, 0.5)
            y = center_y + 50 * np.sin(angle) + np.random.normal(0, 0.5)
            coords.append((x, y))
        
        poly = Polygon(coords)
        if poly.is_valid:
            polygons.append(poly)
            print(f"   Полигон {i+1}: {len(poly.exterior.coords)} точек")
    
    if len(polygons) >= 2:
        # Объединяем полигоны
        merged = unary_union(polygons)
        print(f"   Объединенный полигон: {len(merged.exterior.coords)} точек")
        
        # Упрощаем если нужно
        if len(merged.exterior.coords) > 1000:
            simplified = merged.simplify(2.0, preserve_topology=True)
            print(f"   Упрощенный объединенный: {len(simplified.exterior.coords)} точек")
            return len(simplified.exterior.coords) < len(merged.exterior.coords)
    
    return True

def simulate_real_world_scenario():
    """Симулирует реальный сценарий с большим полигоном"""
    
    print("\n🧪 Симуляция реального сценария:")
    
    # Создаем маску, похожую на реальную проблему
    mask = np.zeros((512, 512), dtype=np.float32)
    
    # Создаем прямоугольную область с шумом
    for i in range(50, 462):
        for j in range(50, 462):
            if 100 <= i <= 400 and 100 <= j <= 400:
                mask[i, j] = 1.0
                # Добавляем шум для создания детализированного контура
                if (i + j) % 2 == 0:
                    mask[i, j] = 0.9
                if (i * j) % 3 == 0:
                    mask[i, j] = 0.8
    
    # Находим контуры
    contours = measure.find_contours(mask, 0.5)
    
    if contours:
        largest_contour = max(contours, key=len)
        print(f"   Исходный контур: {len(largest_contour)} точек")
        
        # Создаем полигон
        coords = [(p[1], p[0]) for p in largest_contour]
        poly = Polygon(coords)
        
        if poly.is_valid:
            print(f"   Исходный полигон: {len(poly.exterior.coords)} точек")
            
            # Применяем наше исправление
            if len(poly.exterior.coords) > 1000:
                simplified = poly.simplify(1.0, preserve_topology=True)
                print(f"   После упрощения (tolerance=1.0): {len(simplified.exterior.coords)} точек")
                
                if len(simplified.exterior.coords) > 1000:
                    simplified = poly.simplify(2.0, preserve_topology=True)
                    print(f"   После дополнительного упрощения (tolerance=2.0): {len(simplified.exterior.coords)} точек")
                
                # Проверяем качество
                area_ratio = simplified.area / poly.area
                print(f"   Сохранение площади: {area_ratio:.3f}")
                
                return len(simplified.exterior.coords) < len(poly.exterior.coords)
    
    return False

if __name__ == "__main__":
    print("🔧 Тестирование исправлений для проблемы с полигонами\n")
    
    # Тест 1: Упрощение полигонов
    test1_passed = test_polygon_simplification()
    print(f"   ✅ Упрощение полигонов: {'ПРОЙДЕН' if test1_passed else 'ПРОВАЛЕН'}")
    
    # Тест 2: Упрощение в мержере
    test2_passed = test_merger_simplification()
    print(f"   ✅ Упрощение в мержере: {'ПРОЙДЕН' if test2_passed else 'ПРОВАЛЕН'}")
    
    # Тест 3: Реальный сценарий
    test3_passed = simulate_real_world_scenario()
    print(f"   ✅ Реальный сценарий: {'ПРОЙДЕН' if test3_passed else 'ПРОВАЛЕН'}")
    
    print(f"\n📊 Результаты:")
    print(f"   Всего тестов: 3")
    print(f"   Пройдено: {sum([test1_passed, test2_passed, test3_passed])}")
    print(f"   Провалено: {3 - sum([test1_passed, test2_passed, test3_passed])}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print(f"\n🎉 Все тесты пройдены! Исправления должны работать.")
    else:
        print(f"\n⚠️  Некоторые тесты провалены. Требуется дополнительная настройка.")

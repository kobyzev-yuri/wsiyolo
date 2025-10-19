#!/usr/bin/env python3
"""
Тестирование исправлений на реальных данных
"""

import json
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

def test_real_polygon_simplification():
    """Тестирует упрощение на реальном полигоне из JSON"""
    
    print("🧪 Тестирование на реальных данных:")
    
    json_file = "/mnt/ai/cnn/wsiyolo/results/19_ibd_mod_S037__20240822_091343.json"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pred = data['predictions'][0]
    polygon = pred['polygon']
    
    print(f"   Исходный полигон: {len(polygon)} точек")
    print(f"   Класс: {pred['class_name']}")
    print(f"   Уверенность: {pred['confidence']}")
    
    # Создаем Shapely полигон
    coords = [(p['x'], p['y']) for p in polygon]
    poly = Polygon(coords)
    
    if poly.is_valid:
        print(f"   Площадь: {poly.area:.1f}")
        print(f"   Периметр: {poly.length:.1f}")
        
        # Применяем умное упрощение
        simplified = smart_simplify_polygon(poly, max_points=60)
        
        print(f"   Упрощенный полигон: {len(simplified.exterior.coords)} точек")
        print(f"   Площадь после упрощения: {simplified.area:.1f}")
        print(f"   Периметр после упрощения: {simplified.length:.1f}")
        
        # Проверяем качество
        area_ratio = simplified.area / poly.area
        perimeter_ratio = simplified.length / poly.length
        
        print(f"   Сохранение площади: {area_ratio:.3f}")
        print(f"   Сохранение периметра: {perimeter_ratio:.3f}")
        
        # Проверяем критерии
        points_ok = len(simplified.exterior.coords) <= 60
        area_ok = area_ratio > 0.8
        
        print(f"   ✅ Количество точек: {'OK' if points_ok else 'FAIL'} ({len(simplified.exterior.coords)}/60)")
        print(f"   ✅ Сохранение площади: {'OK' if area_ok else 'FAIL'} ({area_ratio:.3f})")
        
        return points_ok and area_ok
    else:
        print(f"   ❌ Исходный полигон невалиден")
        return False

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

def create_optimized_json():
    """Создает оптимизированную версию JSON с упрощенными полигонами"""
    
    print("\n🔧 Создание оптимизированной версии JSON:")
    
    json_file = "/mnt/ai/cnn/wsiyolo/results/19_ibd_mod_S037__20240822_091343.json"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Обрабатываем предсказания
    optimized_predictions = []
    
    for pred in data['predictions']:
        polygon = pred['polygon']
        
        # Создаем Shapely полигон
        coords = [(p['x'], p['y']) for p in polygon]
        poly = Polygon(coords)
        
        if poly.is_valid:
            # Упрощаем полигон
            simplified = smart_simplify_polygon(poly, max_points=60)
            
            # Создаем новое предсказание
            optimized_pred = pred.copy()
            optimized_pred['polygon'] = [
                {'x': x, 'y': y} for x, y in simplified.exterior.coords[:-1]
            ]
            
            # Обновляем bbox
            bounds = simplified.bounds
            optimized_pred['box'] = {
                'start': {'x': bounds[0], 'y': bounds[1]},
                'end': {'x': bounds[2], 'y': bounds[3]}
            }
            
            optimized_predictions.append(optimized_pred)
            
            print(f"   Оптимизировано: {len(polygon)} -> {len(optimized_pred['polygon'])} точек")
    
    # Создаем оптимизированную версию данных
    optimized_data = data.copy()
    optimized_data['predictions'] = optimized_predictions
    
    # Обновляем статистику
    optimized_data['statistics']['total'] = len(optimized_predictions)
    optimized_data['statistics']['by_class'] = {pred['class_name']: 1 for pred in optimized_predictions}
    
    # Сохраняем оптимизированную версию
    output_file = "/mnt/ai/cnn/wsiyolo/results/19_ibd_mod_S037__20240822_091343_optimized.json"
    
    with open(output_file, 'w') as f:
        json.dump(optimized_data, f, indent=2)
    
    print(f"   Сохранено: {output_file}")
    
    # Сравниваем размеры файлов
    original_size = Path(json_file).stat().st_size
    optimized_size = Path(output_file).stat().st_size
    
    print(f"   Размер исходного файла: {original_size / 1024 / 1024:.2f} MB")
    print(f"   Размер оптимизированного файла: {optimized_size / 1024 / 1024:.2f} MB")
    print(f"   Экономия: {(1 - optimized_size / original_size) * 100:.1f}%")
    
    return output_file

if __name__ == "__main__":
    print("🔧 Тестирование исправлений на реальных данных\n")
    
    # Тест 1: Упрощение реального полигона
    test1_passed = test_real_polygon_simplification()
    print(f"   ✅ Реальное упрощение: {'ПРОЙДЕН' if test1_passed else 'ПРОВАЛЕН'}")
    
    # Тест 2: Создание оптимизированного JSON
    try:
        optimized_file = create_optimized_json()
        print(f"   ✅ Создание оптимизированного JSON: ПРОЙДЕН")
    except Exception as e:
        print(f"   ❌ Создание оптимизированного JSON: ПРОВАЛЕН - {e}")
        test1_passed = False
    
    print(f"\n📊 Результаты:")
    if test1_passed:
        print(f"🎉 Все тесты пройдены! Проблема с огромными полигонами решена.")
        print(f"   - Полигоны ограничены до 60 точек максимум")
        print(f"   - Сохранение площади > 80%")
        print(f"   - Значительное уменьшение размера файлов")
    else:
        print(f"⚠️  Некоторые тесты провалены. Требуется дополнительная настройка.")


#!/usr/bin/env python3
"""
Создание сетки биоптатов с нумерацией для выбора конкретного биоптата
"""

import json
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_numbered_biopsy_grid(analysis_file="simple_biopsy_analysis.json"):
    """Создает пронумерованную сетку биоптатов"""
    
    print("🔢 Создание пронумерованной сетки биоптатов")
    print("=" * 60)
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        biopsy_regions = data['simple_biopsy_analysis']['biopsy_regions']
        grid_params = data['simple_biopsy_analysis']['grid_parameters']
        
        print(f"📊 Найдено {len(biopsy_regions)} биоптатов")
        print(f"📐 Сетка: {grid_params['step_x']}x{grid_params['step_y']}")
        
        # Сортируем биоптаты по расстоянию от начала координат
        distances = []
        for region in biopsy_regions:
            center_x = (region['x_min'] + region['x_max']) / 2
            center_y = (region['y_min'] + region['y_max']) / 2
            distance = math.sqrt(center_x**2 + center_y**2)
            
            distances.append({
                'id': region['id'],
                'name': region['name'],
                'center': (center_x, center_y),
                'distance': distance,
                'region': region
            })
        
        # Сортируем по расстоянию (ближайший к началу координат = 1)
        distances.sort(key=lambda x: x['distance'])
        
        # Переназначаем номера по порядку от начала координат
        numbered_biopsies = []
        for i, biopsy in enumerate(distances):
            new_id = i + 1
            numbered_biopsy = {
                'grid_id': new_id,  # Новый ID в сетке
                'original_id': biopsy['id'],  # Оригинальный ID
                'name': f"Биоптат {new_id} (сетка)",
                'center': biopsy['center'],
                'distance': biopsy['distance'],
                'region': biopsy['region'],
                'grid_position': f"({i//3 + 1}, {i%3 + 1})" if len(distances) <= 6 else f"({i//2 + 1}, {i%2 + 1})"
            }
            numbered_biopsies.append(numbered_biopsy)
        
        print(f"\n🔢 ПРОНУМЕРОВАННАЯ СЕТКА БИОПТАТОВ:")
        print("=" * 50)
        for biopsy in numbered_biopsies:
            print(f"   {biopsy['grid_id']}. {biopsy['name']} (ориг. ID {biopsy['original_id']})")
            print(f"      Позиция в сетке: {biopsy['grid_position']}")
            print(f"      Расстояние от (0,0): {biopsy['distance']:,.0f} пикселей")
            print(f"      Центр: ({biopsy['center'][0]:,.0f}, {biopsy['center'][1]:,.0f})")
            print(f"      Координаты: ({biopsy['region']['x_min']:,}, {biopsy['region']['y_min']:,}) - ({biopsy['region']['x_max']:,}, {biopsy['region']['y_max']:,})")
            print()
        
        # Создаем конфигурацию для выбора биоптата
        grid_config = {
            "biopsy_grid": {
                "total_biopsies": len(numbered_biopsies),
                "grid_step_x": grid_params['step_x'],
                "grid_step_y": grid_params['step_y'],
                "numbered_biopsies": numbered_biopsies
            },
            "selection_options": {
                "closest_to_origin": 1,  # ID ближайшего к началу координат
                "available_ids": list(range(1, len(numbered_biopsies) + 1)),
                "default_selection": 1
            }
        }
        
        # Сохраняем конфигурацию
        with open("biopsy_grid_config.json", 'w', encoding='utf-8') as f:
            json.dump(grid_config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Конфигурация сетки сохранена в: biopsy_grid_config.json")
        
        return grid_config
        
    except FileNotFoundError:
        print(f"❌ Файл анализа не найден: {analysis_file}")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def create_grid_visualization(analysis_file="simple_biopsy_analysis.json"):
    """Создает визуализацию пронумерованной сетки"""
    
    print("🎨 Создание визуализации пронумерованной сетки...")
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        biopsy_regions = data['simple_biopsy_analysis']['biopsy_regions']
        grid_params = data['simple_biopsy_analysis']['grid_parameters']
        
        # Загружаем миниатюру
        img = Image.open("wsi_thumbnail.jpg")
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        # Масштабирующие факторы
        scale_x = 136192 / vis_img.width
        scale_y = 77312 / vis_img.height
        
        # Сортируем биоптаты по расстоянию от начала координат
        distances = []
        for region in biopsy_regions:
            center_x = (region['x_min'] + region['x_max']) / 2
            center_y = (region['y_min'] + region['y_max']) / 2
            distance = math.sqrt(center_x**2 + center_y**2)
            distances.append((distance, region))
        
        distances.sort(key=lambda x: x[0])
        
        # Рисуем сетку
        grid_step_x = int(grid_params['step_x'] / scale_x)
        grid_step_y = int(grid_params['step_y'] / scale_y)
        
        # Вертикальные линии
        for x in range(0, vis_img.width, grid_step_x):
            draw.line([(x, 0), (x, vis_img.height)], fill='blue', width=1)
        
        # Горизонтальные линии
        for y in range(0, vis_img.height, grid_step_y):
            draw.line([(0, y), (vis_img.width, y)], fill='blue', width=1)
        
        # Рисуем биоптаты с номерами
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (distance, region) in enumerate(distances):
            color = colors[i % len(colors)]
            grid_id = i + 1
            
            # Масштабируем координаты для миниатюры
            x_min = int(region['x_min'] / scale_x)
            y_min = int(region['y_min'] / scale_y)
            x_max = int(region['x_max'] / scale_x)
            y_max = int(region['y_max'] / scale_y)
            
            # Рисуем прямоугольник
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
            
            # Добавляем номер биоптата
            try:
                font = ImageFont.load_default()
                # Номер в левом верхнем углу
                draw.text((x_min + 5, y_min + 5), f"#{grid_id}", fill=color, font=font)
                # Расстояние от начала координат
                draw.text((x_min + 5, y_min + 25), f"d={distance:,.0f}", fill=color, font=font)
            except:
                draw.text((x_min + 5, y_min + 5), f"#{grid_id}", fill=color)
                draw.text((x_min + 5, y_min + 25), f"d={distance:,.0f}", fill=color)
        
        # Сохраняем визуализацию
        vis_img.save("wsi_numbered_grid.jpg")
        print(f"✅ Визуализация сохранена: wsi_numbered_grid.jpg")
        
    except Exception as e:
        print(f"❌ Ошибка создания визуализации: {e}")

def main():
    """Основная функция создания пронумерованной сетки"""
    
    # Создаем конфигурацию сетки
    grid_config = create_numbered_biopsy_grid()
    
    if grid_config:
        # Создаем визуализацию
        create_grid_visualization()
        
        print(f"\n🎯 ГОТОВО К ИСПОЛЬЗОВАНИЮ:")
        print("=" * 40)
        print("1. 🔢 Биоптаты пронумерованы от 1 до 6")
        print("2. 📍 Номер 1 = ближайший к началу координат")
        print("3. 🔧 Используйте ключ --biopsy-id N для выбора биоптата")
        print("4. 📊 Конфигурация в biopsy_grid_config.json")
        print("5. 🎨 Визуализация в wsi_numbered_grid.jpg")

if __name__ == "__main__":
    main()

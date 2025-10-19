#!/usr/bin/env python3
"""
🧬 Единый Workflow для работы с биопсиями

Этот скрипт объединяет все этапы работы с биопсиями:
1. Детекция биопсий
2. Создание пронумерованной сетки
3. Выбор биопсии для обработки
4. Запуск обработки

Автор: WSIYOLO Team
Версия: 1.0
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess

def run_biopsy_detection(wsi_path, output_dir="biopsy_results"):
    """Запускает детекцию биопсий"""
    print("🔍 Этап 1: Детекция биопсий")
    print("-" * 30)
    
    cmd = [
        "python", "detect_biopsies.py",
        "--wsi-path", wsi_path,
        "--output-dir", output_dir,
        "--n-biopsies", "6"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Детекция биопсий завершена успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка детекции биопсий: {e}")
        print(f"Вывод: {e.stdout}")
        print(f"Ошибки: {e.stderr}")
        return False

def create_numbered_grid(results_path, output_dir="biopsy_results"):
    """Создает пронумерованную сетку биопсий"""
    print("\n🔢 Этап 2: Создание пронумерованной сетки")
    print("-" * 40)
    
    try:
        # Загружаем результаты детекции
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        biopsy_regions = data["biopsy_detection"]["biopsy_regions"]
        
        # Сортируем по расстоянию от начала координат
        def distance_from_origin(biopsy):
            center_x = (biopsy["x_min"] + biopsy["x_max"]) / 2
            center_y = (biopsy["y_min"] + biopsy["y_max"]) / 2
            return (center_x ** 2 + center_y ** 2) ** 0.5
        
        sorted_biopsies = sorted(biopsy_regions, key=distance_from_origin)
        
        # Создаем пронумерованную сетку
        numbered_biopsies = []
        for i, biopsy in enumerate(sorted_biopsies, 1):
            numbered_biopsy = {
                "grid_id": i,
                "original_id": biopsy["id"],
                "name": f"Биоптат {i} (сетка)",
                "center": [
                    (biopsy["x_min"] + biopsy["x_max"]) / 2,
                    (biopsy["y_min"] + biopsy["y_max"]) / 2
                ],
                "distance": distance_from_origin(biopsy),
                "region": biopsy
            }
            numbered_biopsies.append(numbered_biopsy)
        
        # Сохраняем конфигурацию сетки
        grid_config = {
            "biopsy_grid": {
                "total_biopsies": len(numbered_biopsies),
                "numbered_biopsies": numbered_biopsies
            },
            "selection_options": {
                "closest_to_origin": 1,
                "available_ids": list(range(1, len(numbered_biopsies) + 1)),
                "default_selection": 1
            }
        }
        
        grid_path = os.path.join(output_dir, "biopsy_grid_config.json")
        with open(grid_path, 'w', encoding='utf-8') as f:
            json.dump(grid_config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Сетка создана: {grid_path}")
        print(f"📊 Доступно биопсий: {len(numbered_biopsies)}")
        
        return grid_path
        
    except Exception as e:
        print(f"❌ Ошибка создания сетки: {e}")
        return None

def select_biopsy_for_processing(biopsy_id, grid_path, output_dir="biopsy_results"):
    """Выбирает биопсию для обработки"""
    print(f"\n🎯 Этап 3: Выбор биопсии {biopsy_id}")
    print("-" * 35)
    
    try:
        # Загружаем конфигурацию сетки
        with open(grid_path, 'r', encoding='utf-8') as f:
            grid_data = json.load(f)
        
        numbered_biopsies = grid_data["biopsy_grid"]["numbered_biopsies"]
        
        # Находим выбранную биопсию
        selected_biopsy = None
        for biopsy in numbered_biopsies:
            if biopsy["grid_id"] == biopsy_id:
                selected_biopsy = biopsy
                break
        
        if not selected_biopsy:
            print(f"❌ Биопсия {biopsy_id} не найдена")
            return None
        
        # Создаем конфигурацию для обработки
        processing_config = {
            "selected_biopsy": selected_biopsy,
            "processing_region": selected_biopsy["region"],
            "grid_info": {
                "total_biopsies": len(numbered_biopsies),
                "selected_id": biopsy_id
            }
        }
        
        config_path = os.path.join(output_dir, f"selected_biopsy_{biopsy_id}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(processing_config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Биопсия {biopsy_id} выбрана для обработки")
        print(f"📍 Регион: ({selected_biopsy['region']['x_min']}, {selected_biopsy['region']['y_min']}) - ({selected_biopsy['region']['x_max']}, {selected_biopsy['region']['y_max']})")
        print(f"📁 Конфигурация: {config_path}")
        
        return config_path
        
    except Exception as e:
        print(f"❌ Ошибка выбора биопсии: {e}")
        return None

def run_biopsy_processing(biopsy_id, wsi_path, config_path, output_dir="biopsy_results"):
    """Запускает обработку выбранной биопсии"""
    print(f"\n🚀 Этап 4: Обработка биопсии {biopsy_id}")
    print("-" * 40)
    
    try:
        # Загружаем конфигурацию
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        biopsy_region = config["selected_biopsy"]["region"]
        
        print(f"📍 Обрабатываемый регион:")
        print(f"   X: {biopsy_region['x_min']} - {biopsy_region['x_max']}")
        print(f"   Y: {biopsy_region['y_min']} - {biopsy_region['y_max']}")
        print(f"   Размер: {biopsy_region['width']} x {biopsy_region['height']}")
        
        # Здесь можно интегрировать с основным pipeline
        # Пока что создаем заглушку
        processing_results = {
            "biopsy_id": biopsy_id,
            "region": biopsy_region,
            "status": "ready_for_processing",
            "message": "Готово для интеграции с основным pipeline"
        }
        
        results_path = os.path.join(output_dir, f"biopsy_{biopsy_id}_processing_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(processing_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Обработка биопсии {biopsy_id} подготовлена")
        print(f"📁 Результаты: {results_path}")
        
        return results_path
        
    except Exception as e:
        print(f"❌ Ошибка обработки биопсии: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Единый workflow для работы с биопсиями")
    parser.add_argument("--wsi-path", required=True, help="Путь к WSI файлу")
    parser.add_argument("--biopsy-id", type=int, default=1, help="ID биопсии для обработки (1-6)")
    parser.add_argument("--output-dir", default="biopsy_results", help="Папка для результатов")
    parser.add_argument("--skip-detection", action="store_true", help="Пропустить детекцию (если уже выполнена)")
    
    args = parser.parse_args()
    
    print("🧬 Единый Workflow для работы с биопсиями")
    print("=" * 50)
    print(f"📁 WSI: {args.wsi_path}")
    print(f"🎯 Биопсия: {args.biopsy_id}")
    print(f"📂 Результаты: {args.output_dir}")
    print()
    
    # Создаем папку для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Этап 1: Детекция биопсий
    if not args.skip_detection:
        if not run_biopsy_detection(args.wsi_path, args.output_dir):
            print("❌ Детекция биопсий не удалась")
            return False
    
    # Этап 2: Создание сетки
    results_path = os.path.join(args.output_dir, "biopsy_detection_results.json")
    if not os.path.exists(results_path):
        print(f"❌ Файл результатов не найден: {results_path}")
        return False
    
    grid_path = create_numbered_grid(results_path, args.output_dir)
    if not grid_path:
        print("❌ Создание сетки не удалось")
        return False
    
    # Этап 3: Выбор биопсии
    config_path = select_biopsy_for_processing(args.biopsy_id, grid_path, args.output_dir)
    if not config_path:
        print("❌ Выбор биопсии не удался")
        return False
    
    # Этап 4: Обработка биопсии
    processing_path = run_biopsy_processing(args.biopsy_id, args.wsi_path, config_path, args.output_dir)
    if not processing_path:
        print("❌ Обработка биопсии не удалась")
        return False
    
    print("\n🎉 Workflow завершен успешно!")
    print(f"📁 Все результаты в папке: {args.output_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

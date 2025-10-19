#!/usr/bin/env python3
"""
Запуск обработки конкретного биоптата с ключом --biopsy-id
"""

import argparse
import json
import sys
import os
from pathlib import Path

def load_biopsy_config(biopsy_id):
    """Загружает конфигурацию выбранного биоптата"""
    
    config_file = f"selected_biopsy_{biopsy_id}_config.json"
    
    if not os.path.exists(config_file):
        print(f"❌ Конфигурация биоптата {biopsy_id} не найдена")
        print(f"💡 Запустите: python select_biopsy_for_processing.py --biopsy-id {biopsy_id}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return None

def run_improved_pipeline_with_biopsy(biopsy_id, wsi_path, output_dir):
    """Запускает улучшенный pipeline для конкретного биоптата"""
    
    print(f"🚀 Запуск улучшенного pipeline для биоптата {biopsy_id}")
    print("=" * 60)
    
    # Загружаем конфигурацию биоптата
    config = load_biopsy_config(biopsy_id)
    if not config:
        return False
    
    selected_biopsy = config['selected_biopsy']
    processing_region = config['processing_region']
    
    print(f"📊 Конфигурация биоптата:")
    print(f"   ID: {selected_biopsy['grid_id']}")
    print(f"   Название: {selected_biopsy['name']}")
    print(f"   Координаты: ({processing_region['x_min']:,}, {processing_region['y_min']:,}) - ({processing_region['x_max']:,}, {processing_region['y_max']:,})")
    print(f"   Размер: {processing_region['width']}x{processing_region['height']}")
    print(f"   Ускорение: в {config['optimization']['speedup_factor']} раз")
    
    # Импортируем улучшенный pipeline
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from improved_wsi_yolo_pipeline import ImprovedWSIYOLOPipeline
    except ImportError as e:
        print(f"❌ Ошибка импорта улучшенного pipeline: {e}")
        return False
    
    # Создаем конфигурацию для pipeline
    pipeline_config = {
        "wsi_path": wsi_path,
        "output_dir": output_dir,
        "biopsy_region": processing_region,
        "models": {
            "lp": "models/lp.pt",
            "mild": "models/mild.pt", 
            "moderate": "models/moderate.pt"
        },
        "performance": {
            "batch_size": 16,
            "max_workers": 4,
            "device": "cuda"
        },
        "optimization": {
            "process_only_biopsy": True,
            "biopsy_id": biopsy_id,
            "speedup_factor": config['optimization']['speedup_factor']
        }
    }
    
    print(f"\n🔧 Запуск pipeline...")
    print(f"   WSI: {wsi_path}")
    print(f"   Выходная папка: {output_dir}")
    print(f"   Область обработки: биоптат {biopsy_id}")
    
    try:
        # Создаем pipeline
        pipeline = ImprovedWSIYOLOPipeline(
            batch_size=pipeline_config['performance']['batch_size'],
            max_workers=pipeline_config['performance']['max_workers'],
            device=pipeline_config['performance']['device']
        )
        
        # Запускаем обработку
        results = pipeline.process_wsi(
            wsi_path=wsi_path,
            output_dir=output_dir,
            biopsy_region=processing_region
        )
        
        print(f"\n✅ Обработка завершена успешно!")
        print(f"📊 Результаты:")
        print(f"   Предсказаний: {len(results.get('predictions', []))}")
        print(f"   Время обработки: {results.get('processing_time', 'N/A')}")
        print(f"   Ускорение: в {config['optimization']['speedup_factor']} раз")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        return False

def main():
    """Основная функция запуска обработки биоптата"""
    
    parser = argparse.ArgumentParser(description='Обработка конкретного биоптата')
    parser.add_argument('--biopsy-id', type=int, required=True, help='ID биоптата для обработки (1-6)')
    parser.add_argument('--wsi-path', default='wsi/19_ibd_mod_S037__20240822_091343.tiff', help='Путь к WSI файлу')
    parser.add_argument('--output-dir', default=f'results_biopsy_', help='Папка для результатов')
    
    args = parser.parse_args()
    
    # Создаем папку для результатов
    output_dir = f"{args.output_dir}{args.biopsy_id}"
    
    print("🚀 Обработка конкретного биоптата")
    print("=" * 50)
    print(f"🔧 Биоптат ID: {args.biopsy_id}")
    print(f"📁 WSI: {args.wsi_path}")
    print(f"📁 Результаты: {output_dir}")
    
    # Проверяем наличие WSI
    if not os.path.exists(args.wsi_path):
        print(f"❌ WSI файл не найден: {args.wsi_path}")
        return 1
    
    # Запускаем обработку
    success = run_improved_pipeline_with_biopsy(args.biopsy_id, args.wsi_path, output_dir)
    
    if success:
        print(f"\n🎯 ОБРАБОТКА ЗАВЕРШЕНА:")
        print("=" * 30)
        print(f"1. ✅ Обработан биоптат ID {args.biopsy_id}")
        print(f"2. 📁 Результаты в: {output_dir}")
        print(f"3. 🚀 Ускорение в 6 раз")
        print(f"4. 📊 Координаты относительно WSI")
        return 0
    else:
        print(f"\n❌ Обработка завершилась с ошибкой")
        return 1

if __name__ == "__main__":
    sys.exit(main())

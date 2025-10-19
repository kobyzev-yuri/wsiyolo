#!/usr/bin/env python3
"""
Запуск WSI YOLO Pipeline
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.wsi_yolo_pipeline import WSIYOLOPipeline, create_models_config


def main():
    """Основная функция для запуска pipeline"""
    
    print("🚀 Запуск WSI YOLO Pipeline")
    print("=" * 50)
    
    # Конфигурация
    models_dir = "models"
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    output_path = "results/predictions.json"
    
    # Проверяем наличие файлов
    if not os.path.exists(models_dir):
        print(f"❌ Директория с моделями не найдена: {models_dir}")
        return False
    
    if not os.path.exists(wsi_path):
        print(f"❌ WSI файл не найден: {wsi_path}")
        return False
    
    try:
        # Создаем конфигурацию моделей
        print("📁 Создание конфигурации моделей...")
        models_config = create_models_config(models_dir)
        print(f"   Найдено моделей: {len(models_config)}")
        
        for config in models_config:
            print(f"   - {config['model_path']}: {config['window_size']}x{config['window_size']}, conf={config['min_conf']}")
        
        # Создаем pipeline
        print("\n🔧 Инициализация pipeline...")
        pipeline = WSIYOLOPipeline(
            models_config=models_config,
            tile_size=512,
            overlap_ratio=0.5,
            iou_threshold=0.5
        )
        
        # Обрабатываем WSI
        print(f"\n🔍 Обработка WSI: {wsi_path}")
        predictions = pipeline.process_wsi(wsi_path, output_path)
        
        # Выводим статистику
        stats = pipeline.get_statistics(predictions)
        print(f"\n📊 Финальная статистика:")
        print(f"   Всего предсказаний: {stats['total']}")
        print(f"   Средняя уверенность: {stats['average_confidence']:.3f}")
        print(f"   По классам: {stats['by_class']}")
        
        if stats['total'] > 0:
            print(f"\n✅ Pipeline выполнен успешно!")
            print(f"   Результаты сохранены: {output_path}")
        else:
            print(f"\n⚠️  Предсказания не найдены")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


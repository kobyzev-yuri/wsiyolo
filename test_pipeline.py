#!/usr/bin/env python3
"""
Тестовый скрипт для WSI YOLO pipeline
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import WSIYOLOPipeline, create_models_config


def test_pipeline():
    """Тестирует pipeline с имеющимися моделями и WSI"""
    
    print("🧪 Тестирование WSI YOLO Pipeline")
    print("=" * 50)
    
    # Проверяем наличие файлов
    models_dir = "models"
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    
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
        
        # Тестируем загрузку WSI
        print("\n📊 Тестирование загрузки WSI...")
        wsi_info = pipeline.wsi_pipeline.load_wsi(wsi_path)
        print(f"   Размер: {wsi_info.width}x{wsi_info.height}")
        print(f"   Уровни: {wsi_info.levels}")
        print(f"   MPP: {wsi_info.mpp}")
        
        # Тестируем извлечение патчей (только первые 10)
        print("\n🔍 Тестирование извлечения патчей...")
        patches = pipeline.wsi_pipeline.extract_patches(wsi_path)
        print(f"   Найдено патчей: {len(patches)}")
        
        if patches:
            print(f"   Первый патч: {patches[0].patch_id}, позиция: ({patches[0].x}, {patches[0].y})")
            print(f"   Размер патча: {patches[0].image.shape}")
        
        # Тестируем YOLO inference (только на первом патче)
        if patches:
            print("\n🤖 Тестирование YOLO inference...")
            first_patch = patches[0]
            predictions = pipeline.yolo_inference.predict_patch(first_patch)
            print(f"   Предсказаний на первом патче: {len(predictions)}")
            
            for pred in predictions:
                print(f"   - {pred.class_name}: conf={pred.conf:.3f}, box=({pred.box.start.x:.0f},{pred.box.start.y:.0f})-({pred.box.end.x:.0f},{pred.box.end.y:.0f})")
        
        print("\n✅ Тестирование завершено успешно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)


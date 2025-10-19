#!/usr/bin/env python3
"""
Отладочный скрипт для проверки каждого этапа pipeline
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.wsi_yolo_pipeline import WSIYOLOPipeline, create_models_config


def debug_pipeline():
    """Отлаживает каждый этап pipeline"""
    
    print("🔍 Отладка WSI YOLO Pipeline")
    print("=" * 50)
    
    # Конфигурация
    models_dir = "models"
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    
    try:
        # Создаем конфигурацию моделей
        models_config = create_models_config(models_dir)
        print(f"📁 Найдено моделей: {len(models_config)}")
        
        # Создаем pipeline
        pipeline = WSIYOLOPipeline(
            models_config=models_config,
            tile_size=512,
            overlap_ratio=0.5,
            iou_threshold=0.5
        )
        
        # 1. Загружаем информацию о WSI
        print("\n📊 Загрузка информации о WSI...")
        wsi_info = pipeline.patch_loader.load_wsi_info(wsi_path)
        print(f"   Размер: {wsi_info.width}x{wsi_info.height}")
        
        # 2. Извлекаем патчи (только 3 для отладки)
        print("\n🔧 Извлечение патчей...")
        patches = pipeline.patch_loader.extract_patches(wsi_path, max_patches=3)
        print(f"   Найдено патчей: {len(patches)}")
        
        if not patches:
            print("❌ Патчи не найдены")
            return False
        
        # 3. Проверяем каждый патч отдельно
        print("\n🤖 Отладка YOLO инференса...")
        all_predictions = []
        
        for i, patch in enumerate(patches):
            print(f"\n   Патч {i+1}: ID={patch.patch_id}, позиция=({patch.x},{patch.y})")
            print(f"   Размер изображения: {patch.image.shape}")
            print(f"   Тип данных: {patch.image.dtype}")
            print(f"   Диапазон значений: {patch.image.min()} - {patch.image.max()}")
            
            # Проверяем каждую модель отдельно
            for model_path, model_data in pipeline.yolo_inference.loaded_models.items():
                model = model_data['model']
                config = model_data['config']
                class_names = model_data['class_names']
                
                print(f"   🔍 Модель: {Path(model_path).name}")
                print(f"      Классы: {list(class_names.values())}")
                print(f"      Минимальная уверенность: {config.min_conf}")
                
                try:
                    # Выполняем предсказание
                    results = model(patch.image, conf=config.min_conf, verbose=False)
                    
                    print(f"      Результаты: {len(results)}")
                    
                    for j, result in enumerate(results):
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy().astype(int)
                            
                            print(f"         Результат {j+1}: {len(boxes)} детекций")
                            
                            for k, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                                class_name = class_names.get(class_id, f"class_{class_id}")
                                print(f"            {k+1}. {class_name}: conf={conf:.3f}, box={box}")
                                
                                # Создаем предсказание
                                from src.data_structures import Coords, Box, Prediction
                                
                                x1, y1, x2, y2 = box
                                absolute_box = Box(
                                    start=Coords(x=patch.x + x1, y=patch.y + y1),
                                    end=Coords(x=patch.x + x2, y=patch.y + y2)
                                )
                                
                                prediction = Prediction(
                                    class_name=class_name,
                                    box=absolute_box,
                                    conf=float(conf),
                                    polygon=None
                                )
                                
                                all_predictions.append(prediction)
                        else:
                            print(f"         Результат {j+1}: нет детекций")
                
                except Exception as e:
                    print(f"      ❌ Ошибка: {e}")
                    continue
        
        print(f"\n📊 Итого предсказаний до объединения: {len(all_predictions)}")
        
        # Показываем все предсказания
        for i, pred in enumerate(all_predictions):
            print(f"   {i+1}. {pred.class_name}: conf={pred.conf:.3f}")
            print(f"      Box: ({pred.box.start.x:.1f}, {pred.box.start.y:.1f}) - ({pred.box.end.x:.1f}, {pred.box.end.y:.1f})")
        
        # 4. Тестируем объединение
        if all_predictions:
            print(f"\n🔗 Тестирование объединения...")
            merged_predictions = pipeline.polygon_merger.merge_predictions(all_predictions)
            print(f"   После объединения: {len(merged_predictions)}")
            
            for i, pred in enumerate(merged_predictions):
                print(f"   {i+1}. {pred.class_name}: conf={pred.conf:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_pipeline()
    sys.exit(0 if success else 1)



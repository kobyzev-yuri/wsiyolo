#!/usr/bin/env python3
"""
Обработка всех WSI файлов и сбор всех предсказаний по всем меткам.
Сохраняет результаты в wsi_name.json для каждого WSI.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.wsi_yolo_pipeline import WSIYOLOPipeline, create_models_config


def process_all_wsi(wsi_dir: str = "wsi", models_dir: str = "models", 
                   results_dir: str = "results", max_patches: int = None):
    """
    Обрабатывает все WSI файлы в директории
    
    Args:
        wsi_dir: Директория с WSI файлами
        models_dir: Директория с моделями
        results_dir: Директория для результатов
        max_patches: Максимальное количество патчей (для тестирования)
    """
    
    print("🚀 Обработка всех WSI файлов")
    print("=" * 50)
    
    # Проверяем директории
    if not os.path.exists(wsi_dir):
        print(f"❌ Директория WSI не найдена: {wsi_dir}")
        return False
    
    if not os.path.exists(models_dir):
        print(f"❌ Директория моделей не найдена: {models_dir}")
        return False
    
    # Создаем директорию результатов
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Создаем конфигурацию моделей
        print("📁 Создание конфигурации моделей...")
        models_config = create_models_config(models_dir)
        print(f"   Найдено моделей: {len(models_config)}")
        
        # Создаем pipeline
        print("\n🔧 Инициализация pipeline...")
        pipeline = WSIYOLOPipeline(
            models_config=models_config,
            tile_size=512,
            overlap_ratio=0.5,
            iou_threshold=0.5
        )
        
        # Находим все WSI файлы
        wsi_files = []
        for ext in ['*.tiff', '*.tif', '*.svs', '*.ndpi']:
            wsi_files.extend(Path(wsi_dir).glob(ext))
        
        if not wsi_files:
            print(f"❌ WSI файлы не найдены в {wsi_dir}")
            return False
        
        print(f"\n📊 Найдено WSI файлов: {len(wsi_files)}")
        
        # Обрабатываем каждый WSI
        all_results = {}
        
        for i, wsi_file in enumerate(wsi_files, 1):
            print(f"\n🔍 Обработка WSI {i}/{len(wsi_files)}: {wsi_file.name}")
            
            try:
                # Обрабатываем WSI
                predictions = pipeline.process_wsi(
                    str(wsi_file), 
                    None,  # Не сохраняем промежуточные результаты
                    max_patches
                )
                
                # Получаем статистику
                stats = pipeline.get_statistics(predictions)
                
                # Создаем имя файла результата
                wsi_name = wsi_file.stem
                result_file = f"{wsi_name}.json"
                result_path = os.path.join(results_dir, result_file)
                
                # Подготавливаем данные для сохранения
                result_data = {
                    'wsi_info': {
                        'path': str(wsi_file),
                        'name': wsi_name,
                        'file': wsi_file.name
                    },
                    'processing_info': {
                        'models_used': len(models_config),
                        'max_patches': max_patches,
                        'tile_size': 512,
                        'overlap_ratio': 0.5
                    },
                    'statistics': stats,
                    'label_statistics': stats['by_class'],  # Статистика по меткам
                    'predictions': []
                }
                
                # Добавляем предсказания
                for pred in predictions:
                    pred_data = {
                        'class_name': pred.class_name,
                        'confidence': pred.conf,
                        'box': {
                            'start': {'x': pred.box.start.x, 'y': pred.box.start.y},
                            'end': {'x': pred.box.end.x, 'y': pred.box.end.y}
                        }
                    }
                    
                    if pred.polygon:
                        pred_data['polygon'] = [
                            {'x': p.x, 'y': p.y} for p in pred.polygon
                        ]
                    
                    result_data['predictions'].append(pred_data)
                
                # Сохраняем результат
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ Обработано: {stats['total']} предсказаний")
                print(f"   📊 По меткам: {stats['by_class']}")
                print(f"   💾 Сохранено: {result_file}")
                
                # Сохраняем в общий результат
                all_results[wsi_name] = {
                    'file': wsi_file.name,
                    'predictions_count': stats['total'],
                    'label_statistics': stats['by_class'],  # Статистика по меткам
                    'avg_confidence': stats['average_confidence']
                }
                
            except Exception as e:
                print(f"   ❌ Ошибка обработки {wsi_file.name}: {e}")
                continue
        
        # Сохраняем общую сводку
        summary_file = os.path.join(results_dir, "processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Подсчитываем общую статистику по меткам
        total_label_stats = {}
        for result in all_results.values():
            for label, count in result['label_statistics'].items():
                if label not in total_label_stats:
                    total_label_stats[label] = 0
                total_label_stats[label] += count
        
        print(f"\n📊 Общая сводка:")
        print(f"   Обработано WSI: {len(all_results)}")
        print(f"   Всего предсказаний: {sum(r['predictions_count'] for r in all_results.values())}")
        print(f"   📊 Общая статистика по меткам:")
        for label, count in sorted(total_label_stats.items()):
            print(f"      {label}: {count}")
        print(f"   Сводка сохранена: {summary_file}")
        
        # Добавляем общую статистику в сводку
        all_results['_summary'] = {
            'total_wsi': len(all_results),
            'total_predictions': sum(r['predictions_count'] for r in all_results.values()),
            'total_label_statistics': total_label_stats
        }
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Основная функция"""
    
    # Параметры
    wsi_dir = "wsi"
    models_dir = "models"
    results_dir = "results"
    max_patches = None  # Обрабатываем весь WSI
    
    print(f"🔧 Параметры:")
    print(f"   WSI директория: {wsi_dir}")
    print(f"   Модели директория: {models_dir}")
    print(f"   Результаты директория: {results_dir}")
    print(f"   Максимум патчей: {max_patches}")
    
    success = process_all_wsi(wsi_dir, models_dir, results_dir, max_patches)
    
    if success:
        print(f"\n✅ Обработка завершена успешно!")
    else:
        print(f"\n❌ Обработка завершена с ошибками")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

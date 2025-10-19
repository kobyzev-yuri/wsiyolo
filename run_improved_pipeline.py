#!/usr/bin/env python3
"""
Запуск улучшенного WSI YOLO Pipeline с полной обработкой.
Использует оптимизированные параметры для максимальной производительности.
"""

import os
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from improved_wsi_yolo_pipeline import ImprovedWSIYOLOPipeline

def main():
    """Запуск улучшенного pipeline с полной обработкой"""
    print("🚀 Запуск улучшенного WSI YOLO Pipeline")
    print("=" * 60)
    
    # Пути к моделям
    model_paths = {
        'lp': 'models/lp.pt',
        'mild': 'models/mild.pt',
        'moderate': 'models/moderate.pt'
    }
    
    # Проверяем наличие моделей
    missing_models = []
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        print("❌ Отсутствуют модели:")
        for missing in missing_models:
            print(f"   {missing}")
        return False
    
    # Проверяем наличие WSI файла
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    if not os.path.exists(wsi_path):
        print(f"❌ WSI файл не найден: {wsi_path}")
        return False
    
    print("✅ Все необходимые файлы найдены")
    
    # Определяем оптимальную конфигурацию на основе доступной памяти
    import torch
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"🔍 Обнаружена GPU память: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            config = {'batch_size': 8, 'max_workers': 2, 'name': 'Low Memory'}
        elif gpu_memory < 16:
            config = {'batch_size': 16, 'max_workers': 4, 'name': 'Balanced'}
        else:
            config = {'batch_size': 32, 'max_workers': 6, 'name': 'High Performance'}
    else:
        config = {'batch_size': 8, 'max_workers': 2, 'name': 'CPU Only'}
    
    print(f"⚙️  Выбрана конфигурация: {config['name']}")
    print(f"   Батч размер: {config['batch_size']}")
    print(f"   Потоков: {config['max_workers']}")
    
    # Инициализируем pipeline
    pipeline = ImprovedWSIYOLOPipeline(
        model_paths=model_paths,
        patch_size=512,
        batch_size=config['batch_size'],
        max_workers=config['max_workers']
    )
    
    # Запускаем обработку
    print(f"\n🚀 Запуск обработки WSI...")
    results = pipeline.process_wsi(wsi_path, "results_improved_full")
    
    if 'error' in results:
        print(f"❌ Ошибка обработки: {results['error']}")
        return False
    
    # Анализируем результаты
    predictions = results.get('predictions', [])
    print(f"\n📊 Результаты обработки:")
    print(f"   Предсказаний получено: {len(predictions)}")
    
    # Статистика по классам
    class_stats = {}
    for pred in predictions:
        class_name = pred.get('class_name', 'unknown')
        class_stats[class_name] = class_stats.get(class_name, 0) + 1
    
    print(f"   Распределение по классам:")
    for class_name, count in class_stats.items():
        print(f"     {class_name}: {count}")
    
    # Статистика упрощения полигонов
    simplified_count = 0
    total_area_preserved = 0
    for pred in predictions:
        if 'simplification_metrics' in pred:
            simplified_count += 1
            total_area_preserved += pred['simplification_metrics'].get('area_preserved', 1.0)
    
    if simplified_count > 0:
        avg_area_preserved = total_area_preserved / simplified_count
        print(f"   Полигонов упрощено: {simplified_count}")
        print(f"   Среднее сохранение площади: {avg_area_preserved:.1%}")
    
    print(f"\n✅ Обработка завершена успешно!")
    print(f"📁 Результаты сохранены в: results_improved_full/")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n💡 Рекомендации:")
        print("   - Сравните результаты с оригинальным pipeline")
        print("   - Проанализируйте качество предсказаний")
        print("   - Настройте параметры под ваше оборудование")
    else:
        print("\n❌ Обработка завершена с ошибками")
        print("   - Проверьте наличие всех необходимых файлов")
        print("   - Убедитесь в корректности путей")
        print("   - Проверьте доступность GPU/CPU ресурсов")

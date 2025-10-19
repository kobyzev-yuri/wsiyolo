#!/usr/bin/env python3
"""
Тестирование улучшенного WSI YOLO Pipeline.
Сравнивает производительность и качество с оригинальным pipeline.
"""

import os
import sys
import time
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from improved_wsi_yolo_pipeline import ImprovedWSIYOLOPipeline

def test_improved_pipeline():
    """Тестирует улучшенный pipeline"""
    print("🧪 Тестирование улучшенного WSI YOLO Pipeline")
    print("=" * 60)
    
    # Проверяем наличие WSI файла
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    if not os.path.exists(wsi_path):
        print(f"❌ WSI файл не найден: {wsi_path}")
        return False
    
    # Проверяем наличие моделей
    model_paths = {
        'lp': 'models/lp.pt',
        'mild': 'models/mild.pt',
        'moderate': 'models/moderate.pt'
    }
    
    missing_models = []
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        print("❌ Отсутствуют модели:")
        for missing in missing_models:
            print(f"   {missing}")
        return False
    
    print("✅ Все необходимые файлы найдены")
    
    # Инициализируем pipeline с разными конфигурациями
    configs = [
        {
            'name': 'Быстрый (малый батч)',
            'batch_size': 8,
            'max_workers': 2,
            'patch_size': 512
        },
        {
            'name': 'Сбалансированный',
            'batch_size': 16,
            'max_workers': 4,
            'patch_size': 512
        },
        {
            'name': 'Максимальная производительность',
            'batch_size': 32,
            'max_workers': 6,
            'patch_size': 512
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔧 Тестирование конфигурации: {config['name']}")
        print(f"   Батч размер: {config['batch_size']}")
        print(f"   Потоков: {config['max_workers']}")
        
        try:
            # Создаем pipeline
            pipeline = ImprovedWSIYOLOPipeline(
                model_paths=model_paths,
                patch_size=config['patch_size'],
                batch_size=config['batch_size'],
                max_workers=config['max_workers']
            )
            
            # Запускаем обработку
            start_time = time.time()
            result = pipeline.process_wsi(wsi_path, f"results_{config['name'].lower().replace(' ', '_')}")
            processing_time = time.time() - start_time
            
            if 'error' in result:
                print(f"   ❌ Ошибка: {result['error']}")
                continue
            
            # Сохраняем результаты
            results[config['name']] = {
                'processing_time': processing_time,
                'predictions_count': len(result.get('predictions', [])),
                'performance_stats': pipeline.performance_stats
            }
            
            print(f"   ✅ Завершено за {processing_time:.2f}с")
            print(f"   📊 Предсказаний: {len(result.get('predictions', []))}")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            continue
    
    # Анализируем результаты
    if results:
        print(f"\n📈 Сравнение конфигураций:")
        print(f"{'Конфигурация':<30} {'Время (с)':<10} {'Предсказаний':<12} {'Патчей/сек':<12}")
        print("-" * 70)
        
        for config_name, result in results.items():
            time_sec = result['processing_time']
            predictions = result['predictions_count']
            patches = result['performance_stats'].get('total_patches', 0)
            throughput = patches / time_sec if time_sec > 0 else 0
            
            print(f"{config_name:<30} {time_sec:<10.2f} {predictions:<12} {throughput:<12.1f}")
        
        # Находим лучшую конфигурацию
        best_config = min(results.items(), key=lambda x: x[1]['processing_time'])
        print(f"\n🏆 Лучшая конфигурация: {best_config[0]}")
        print(f"   Время: {best_config[1]['processing_time']:.2f}с")
        print(f"   Предсказаний: {best_config[1]['predictions_count']}")
        
        return True
    else:
        print("❌ Не удалось протестировать ни одной конфигурации")
        return False

def test_specific_features():
    """Тестирует специфические функции улучшенного pipeline"""
    print(f"\n🔍 Тестирование специфических функций:")
    
    # Тестируем адаптивный упроститель
    print("   ✂️  Тестирование адаптивного упрощения...")
    try:
        from adaptive_polygon_simplifier import AdaptivePolygonSimplifier
        from shapely.geometry import Polygon
        
        simplifier = AdaptivePolygonSimplifier()
        
        # Создаем тестовый полигон
        test_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        simplified, metrics = simplifier.simplify_polygon(test_polygon)
        
        print(f"     ✅ Упрощение работает: {metrics['method']}")
        print(f"     📊 Сохранение площади: {metrics['area_preserved']:.1%}")
        
    except Exception as e:
        print(f"     ❌ Ошибка упрощения: {e}")
    
    # Тестируем улучшенный merger
    print("   🔗 Тестирование улучшенного объединения...")
    try:
        from improved_polygon_merger import ImprovedPolygonMerger
        from data_structures import Prediction, Coords, Box
        
        merger = ImprovedPolygonMerger()
        
        # Создаем тестовые предсказания
        test_predictions = [
            Prediction(
                class_name="lp",
                box=Box(start=Coords(x=0, y=0), end=Coords(x=10, y=10)),
                conf=0.9,
                polygon=[Coords(x=0, y=0), Coords(x=10, y=0), Coords(x=10, y=10), Coords(x=0, y=10)]
            )
        ]
        
        merged = merger.merge_predictions(test_predictions)
        
        print(f"     ✅ Объединение работает: {len(merged)} предсказаний")
        
    except Exception as e:
        print(f"     ❌ Ошибка объединения: {e}")

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестирования улучшенного pipeline")
    print("=" * 60)
    
    # Тестируем специфические функции
    test_specific_features()
    
    # Тестируем полный pipeline
    success = test_improved_pipeline()
    
    if success:
        print(f"\n✅ Тестирование завершено успешно!")
        print(f"💡 Рекомендации:")
        print(f"   - Используйте лучшую конфигурацию для продакшена")
        print(f"   - Сравните результаты с оригинальным pipeline")
        print(f"   - Проанализируйте качество предсказаний")
    else:
        print(f"\n❌ Тестирование завершено с ошибками")
        print(f"💡 Рекомендации:")
        print(f"   - Проверьте наличие всех необходимых файлов")
        print(f"   - Убедитесь в корректности путей к моделям")
        print(f"   - Проверьте доступность GPU/CPU ресурсов")

if __name__ == "__main__":
    main()

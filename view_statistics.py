#!/usr/bin/env python3
"""
Просмотр статистики по WSI и меткам
"""

import json
import sys
from pathlib import Path


def view_statistics(results_dir: str = "results"):
    """
    Показывает статистику по всем WSI
    
    Args:
        results_dir: Директория с результатами
    """
    
    print("📊 Статистика по WSI и меткам")
    print("=" * 50)
    
    # Загружаем сводку
    summary_file = Path(results_dir) / "processing_summary.json"
    
    if not summary_file.exists():
        print(f"❌ Файл сводки не найден: {summary_file}")
        return False
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # Показываем статистику по каждому WSI
        print(f"\n📁 Обработано WSI: {len(summary)}")
        
        total_predictions = 0
        all_labels = {}
        
        for wsi_name, data in summary.items():
            if wsi_name.startswith('_'):  # Пропускаем служебные записи
                continue
                
            print(f"\n🔍 {wsi_name}:")
            print(f"   Файл: {data['file']}")
            print(f"   Предсказаний: {data['predictions_count']}")
            print(f"   Средняя уверенность: {data['avg_confidence']:.3f}")
            print(f"   📊 По меткам:")
            
            for label, count in data['label_statistics'].items():
                print(f"      {label}: {count}")
                all_labels[label] = all_labels.get(label, 0) + count
            
            total_predictions += data['predictions_count']
        
        # Общая статистика
        print(f"\n📊 Общая статистика:")
        print(f"   Всего WSI: {len(summary)}")
        print(f"   Всего предсказаний: {total_predictions}")
        print(f"   📊 Все метки:")
        for label, count in sorted(all_labels.items()):
            print(f"      {label}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки статистики: {e}")
        return False


def view_wsi_details(wsi_name: str, results_dir: str = "results"):
    """
    Показывает детали по конкретному WSI
    
    Args:
        wsi_name: Имя WSI файла
        results_dir: Директория с результатами
    """
    
    wsi_file = Path(results_dir) / f"{wsi_name}.json"
    
    if not wsi_file.exists():
        print(f"❌ Файл WSI не найден: {wsi_file}")
        return False
    
    try:
        with open(wsi_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"🔍 Детали WSI: {wsi_name}")
        print("=" * 50)
        
        # Информация о WSI
        wsi_info = data['wsi_info']
        print(f"📁 Файл: {wsi_info['file']}")
        print(f"📁 Путь: {wsi_info['path']}")
        
        # Информация об обработке
        proc_info = data['processing_info']
        print(f"\n🔧 Параметры обработки:")
        print(f"   Моделей использовано: {proc_info['models_used']}")
        print(f"   Максимум патчей: {proc_info['max_patches']}")
        print(f"   Размер патча: {proc_info['tile_size']}")
        print(f"   Перекрытие: {proc_info['overlap_ratio']*100:.1f}%")
        
        # Статистика
        stats = data['statistics']
        print(f"\n📊 Статистика:")
        print(f"   Всего предсказаний: {stats['total']}")
        print(f"   Средняя уверенность: {stats['average_confidence']:.3f}")
        print(f"   📊 По меткам:")
        for label, count in stats['by_class'].items():
            print(f"      {label}: {count}")
        
        # Предсказания
        predictions = data['predictions']
        print(f"\n🎯 Предсказания ({len(predictions)}):")
        for i, pred in enumerate(predictions, 1):
            print(f"   {i}. {pred['class_name']} (conf: {pred['confidence']:.3f})")
            box = pred['box']
            print(f"      Box: ({box['start']['x']:.1f}, {box['start']['y']:.1f}) - ({box['end']['x']:.1f}, {box['end']['y']:.1f})")
            if 'polygon' in pred and pred['polygon']:
                print(f"      Полигон: {len(pred['polygon'])} точек")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки деталей WSI: {e}")
        return False


def main():
    """Основная функция"""
    
    if len(sys.argv) > 1:
        # Показать детали конкретного WSI
        wsi_name = sys.argv[1]
        success = view_wsi_details(wsi_name)
    else:
        # Показать общую статистику
        success = view_statistics()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



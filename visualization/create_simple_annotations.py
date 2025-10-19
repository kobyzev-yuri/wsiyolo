#!/usr/bin/env python3
"""
Простое создание аннотаций без openslide
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_predictions(json_path):
    """Загружает предсказания из JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_class_colors():
    """Возвращает цвета для разных классов"""
    classes = [
        'Crypts', 'Muscularis mucosae', 'Surface epithelium', 
        'moderate_segmentation', 'Mild', 'excl'
    ]
    
    colors = [
        '#FF6B6B',  # Красный для Crypts
        '#4ECDC4',  # Бирюзовый для Muscularis mucosae
        '#45B7D1',  # Синий для Surface epithelium
        '#96CEB4',  # Зеленый для moderate_segmentation
        '#FFEAA7',  # Желтый для Mild
        '#DDA0DD'   # Фиолетовый для excl
    ]
    
    return dict(zip(classes, colors))

def create_prediction_visualization(predictions, output_dir="prediction_visualizations", max_predictions=100):
    """Создает визуализацию предсказаний"""
    
    print("🔍 Загрузка предсказаний...")
    data = load_predictions(json_path)
    predictions = data['predictions']
    
    print(f"   Найдено предсказаний: {len(predictions)}")
    
    # Фильтруем предсказания - исключаем только excl
    useful_predictions = [p for p in predictions if p['class_name'] != 'excl']
    print(f"   Полезных предсказаний (без excl): {len(useful_predictions)}")
    
    # Группируем по патчам 512x512
    patch_groups = {}
    for pred in useful_predictions:
        x = pred['box']['start']['x']
        y = pred['box']['start']['y']
        patch_x = (x // 512) * 512
        patch_y = (y // 512) * 512
        patch_key = (patch_x, patch_y)
        
        if patch_key not in patch_groups:
            patch_groups[patch_key] = []
        patch_groups[patch_key].append(pred)
    
    print(f"   Найдено патчей с предсказаниями: {len(patch_groups)}")
    
    # Берем область с предсказаниями (0-20000 по X и Y)
    target_patches = {}
    
    for (patch_x, patch_y), patch_preds in patch_groups.items():
        # Проверяем, что патч в целевой области (0-20000)
        if patch_x < 20000 and patch_y < 20000:
            target_patches[(patch_x, patch_y)] = patch_preds
    
    print(f"   Патчей в области 0-20000: {len(target_patches)}")
    
    # Ограничиваем количество для визуализации
    selected_patches = list(target_patches.items())[:max_predictions]
    
    # Создаем выходную директорию
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Получаем цвета для классов
    class_colors = get_class_colors()
    
    # Создаем визуализацию
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    
    # Рисуем патчи с предсказаниями
    legend_elements = []
    class_counts = {}
    
    for (patch_x, patch_y), patch_predictions in selected_patches:
        # Рисуем границу патча 512x512
        patch_rect = patches.Rectangle((patch_x, patch_y), 512, 512, 
                                     linewidth=2, edgecolor='black', facecolor='none', alpha=0.5)
        ax.add_patch(patch_rect)
        
        # Рисуем предсказания внутри патча
        for pred in patch_predictions:
            class_name = pred['class_name']
            confidence = pred['confidence']
            box = pred['box']
            polygon = pred.get('polygon', [])
            
            color = class_colors.get(class_name, '#000000')
            
            # Подсчитываем классы
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            
            # Рисуем bounding box
            x1, y1 = box['start']['x'], box['start']['y']
            x2, y2 = box['end']['x'], box['end']['y']
            width, height = x2 - x1, y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=1, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Рисуем полигон если есть
            if polygon and len(polygon) >= 3:
                poly_coords = [[point['x'], point['y']] for point in polygon]
                poly = patches.Polygon(poly_coords, linewidth=1, 
                                     edgecolor=color, facecolor=color, alpha=0.3)
                ax.add_patch(poly)
    
    # Настраиваем ось - область с предсказаниями (0-20000)
    ax.set_xlim(0, 20000)  # Область с предсказаниями
    ax.set_ylim(0, 20000)   # Область с предсказаниями
    ax.set_aspect('equal')
    ax.set_title(f'Патчи с предсказаниями (область 0-20000): {len(selected_patches)} патчей', fontsize=16)
    ax.set_xlabel('X координата')
    ax.set_ylabel('Y координата')
    ax.grid(True, alpha=0.3)
    
    # Создаем легенду
    for class_name, count in class_counts.items():
        color = class_colors.get(class_name, '#000000')
        legend_elements.append(patches.Patch(color=color, label=f'{class_name} ({count})'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Сохраняем
    output_file = output_path / "all_predictions_overview.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Визуализация сохранена: {output_file}")
    
    return output_path

def create_class_statistics(json_path, output_dir="prediction_visualizations"):
    """Создает статистику по классам"""
    
    print("📊 Создание статистики по классам...")
    
    data = load_predictions(json_path)
    predictions = data['predictions']
    
    # Подсчитываем статистику по классам
    class_counts = {}
    class_confidences = {}
    class_areas = {}
    
    for pred in predictions:
        class_name = pred['class_name']
        confidence = pred['confidence']
        box = pred['box']
        
        # Площадь bbox
        width = box['end']['x'] - box['start']['x']
        height = box['end']['y'] - box['start']['y']
        area = width * height
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
            class_areas[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
        class_areas[class_name].append(area)
    
    # Создаем графики
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # График 1: Количество предсказаний по классам
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [get_class_colors().get(cls, '#000000') for cls in classes]
    
    bars = ax1.bar(classes, counts, color=colors, alpha=0.7)
    ax1.set_title('Количество предсказаний по классам')
    ax1.set_ylabel('Количество')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # График 2: Распределение уверенности
    for class_name, confidences in class_confidences.items():
        ax2.hist(confidences, alpha=0.6, label=class_name, 
                color=get_class_colors().get(class_name, '#000000'), bins=20)
    
    ax2.set_title('Распределение уверенности по классам')
    ax2.set_xlabel('Уверенность')
    ax2.set_ylabel('Частота')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График 3: Распределение площадей
    for class_name, areas in class_areas.items():
        ax3.hist(areas, alpha=0.6, label=class_name, 
                color=get_class_colors().get(class_name, '#000000'), bins=20)
    
    ax3.set_title('Распределение площадей по классам')
    ax3.set_xlabel('Площадь (пиксели²)')
    ax3.set_ylabel('Частота')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # График 4: Средние значения
    mean_confidences = [np.mean(class_confidences[cls]) for cls in classes]
    mean_areas = [np.mean(class_areas[cls]) for cls in classes]
    
    ax4.scatter(mean_areas, mean_confidences, c=colors, s=100, alpha=0.7)
    ax4.set_title('Средняя уверенность vs Средняя площадь')
    ax4.set_xlabel('Средняя площадь')
    ax4.set_ylabel('Средняя уверенность')
    ax4.grid(True, alpha=0.3)
    
    # Добавляем подписи
    for i, cls in enumerate(classes):
        ax4.annotate(cls, (mean_areas[i], mean_confidences[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    stats_file = output_path / "class_statistics.png"
    plt.savefig(stats_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Статистика сохранена: {stats_file}")
    
    return stats_file

if __name__ == "__main__":
    json_path = "results/predictions.json"
    
    print("🎨 Создание визуализации предсказаний")
    print("=" * 50)
    
    # Создаем общую визуализацию
    output_dir = create_prediction_visualization(json_path, max_predictions=500)
    
    # Создаем статистику
    create_class_statistics(json_path, str(output_dir))
    
    print(f"\n🎉 Готово! Результаты в директории: {output_dir}")

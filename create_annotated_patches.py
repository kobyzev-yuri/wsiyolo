#!/usr/bin/env python3
"""
Создание аннотированных патчей по предсказаниям
"""

import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random

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
    
    # Создаем яркие цвета для каждого класса
    colors = [
        '#FF6B6B',  # Красный для Crypts
        '#4ECDC4',  # Бирюзовый для Muscularis mucosae
        '#45B7D1',  # Синий для Surface epithelium
        '#96CEB4',  # Зеленый для moderate_segmentation
        '#FFEAA7',  # Желтый для Mild
        '#DDA0DD'   # Фиолетовый для excl
    ]
    
    return dict(zip(classes, colors))

def create_annotated_patch(patch_image, predictions, patch_coords, class_colors):
    """Создает аннотированный патч"""
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Оригинальный патч
    ax1.imshow(patch_image)
    ax1.set_title('Оригинальный патч')
    ax1.axis('off')
    
    # Аннотированный патч
    ax2.imshow(patch_image)
    ax2.set_title('Аннотированный патч')
    ax2.axis('off')
    
    # Добавляем предсказания
    legend_elements = []
    
    for pred in predictions:
        class_name = pred['class_name']
        confidence = pred['confidence']
        box = pred['box']
        polygon = pred.get('polygon', [])
        
        color = class_colors.get(class_name, '#000000')
        
        # Рисуем bounding box
        x1, y1 = box['start']['x'] - patch_coords[0], box['start']['y'] - patch_coords[1]
        x2, y2 = box['end']['x'] - patch_coords[0], box['end']['y'] - patch_coords[1]
        width, height = x2 - x1, y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
        ax2.add_patch(rect)
        
        # Рисуем полигон если есть
        if polygon:
            poly_coords = []
            for point in polygon:
                px = point['x'] - patch_coords[0]
                py = point['y'] - patch_coords[1]
                poly_coords.append([px, py])
            
            if len(poly_coords) >= 3:
                poly = patches.Polygon(poly_coords, linewidth=1, 
                                     edgecolor=color, facecolor=color, alpha=0.3)
                ax2.add_patch(poly)
        
        # Добавляем в легенду
        legend_elements.append(patches.Patch(color=color, 
                                           label=f'{class_name} ({confidence:.2f})'))
    
    # Добавляем легенду
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig

def extract_patch_from_wsi(wsi_path, x, y, size=512):
    """Извлекает патч из WSI"""
    try:
        import openslide
        slide = openslide.OpenSlide(wsi_path)
        
        # Извлекаем патч на уровне 0 (полное разрешение)
        patch = slide.read_region((int(x), int(y)), 0, (size, size))
        patch = np.array(patch.convert('RGB'))
        
        slide.close()
        return patch
    except ImportError:
        print("⚠️  openslide не установлен, пытаемся установить...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "openslide-python"], check=True)
            print("✅ openslide установлен, перезапустите скрипт")
        except:
            print("❌ Не удалось установить openslide")
        return None
    except Exception as e:
        print(f"⚠️  Ошибка загрузки WSI: {e}")
        return None

def group_predictions_by_patches(predictions):
    """Группирует предсказания по патчам"""
    patch_groups = {}
    
    for pred in predictions:
        # Определяем патч по координатам bbox
        x = pred['box']['start']['x']
        y = pred['box']['start']['y']
        
        # Округляем до размера патча (512)
        patch_x = (x // 512) * 512
        patch_y = (y // 512) * 512
        patch_key = (patch_x, patch_y)
        
        if patch_key not in patch_groups:
            patch_groups[patch_key] = []
        patch_groups[patch_key].append(pred)
    
    return patch_groups

def get_patch_grid_coordinates(patch_x, patch_y, patch_size=512):
    """Получает координаты патча в сетке (i, j)"""
    i = int(patch_x // patch_size)
    j = int(patch_y // patch_size)
    return i, j

def get_wsi_base_name(wsi_path):
    """Получает базовое имя WSI файла без расширения"""
    from pathlib import Path
    return Path(wsi_path).stem

def create_annotated_patches(json_path, wsi_path, output_dir="annotated_patches", max_patches=None):
    """Создает аннотированные патчи"""
    
    print("🔍 Загрузка предсказаний...")
    data = load_predictions(json_path)
    predictions = data['predictions']
    
    print(f"   Найдено предсказаний: {len(predictions)}")
    
    # Фильтруем предсказания - исключаем только excl
    useful_predictions = [p for p in predictions if p['class_name'] != 'excl']
    print(f"   Полезных предсказаний (без excl): {len(useful_predictions)}")
    
    # Создаем выходную директорию
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Получаем цвета для классов
    class_colors = get_class_colors()
    
    # Группируем предсказания по патчам
    print("📊 Группировка предсказаний по патчам...")
    patch_groups = group_predictions_by_patches(useful_predictions)
    
    print(f"   Найдено патчей с полезными предсказаниями: {len(patch_groups)}")
    
    # Сортируем по количеству предсказаний
    sorted_patches = sorted(patch_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Если max_patches не указан, берем все патчи
    if max_patches is None:
        selected_patches = sorted_patches
        print(f"   Создаем аннотированные патчи для ВСЕХ патчей с предсказаниями: {len(selected_patches)}")
    else:
        selected_patches = sorted_patches[:max_patches]
        print(f"   Выбрано топ-{len(selected_patches)} патчей с наибольшим количеством предсказаний")
    
    print(f"🔧 Создание аннотированных патчей...")
    
    # Получаем базовое имя WSI
    wsi_base_name = get_wsi_base_name(wsi_path)
    
    for i, ((patch_x, patch_y), patch_predictions) in enumerate(selected_patches):
        # Получаем координаты патча в сетке
        patch_i, patch_j = get_patch_grid_coordinates(patch_x, patch_y)
        
        print(f"   Обработка патча {i+1}/{len(selected_patches)}: ({patch_x}, {patch_y}) -> grid({patch_i}, {patch_j}) - {len(patch_predictions)} предсказаний")
        
        # Извлекаем патч из WSI
        patch_image = extract_patch_from_wsi(wsi_path, patch_x, patch_y)
        
        if patch_image is None:
            print(f"     ⚠️  Пропущен патч {i+1}: не удалось загрузить изображение")
            continue
        
        # Создаем аннотированный патч
        fig = create_annotated_patch(patch_image, patch_predictions, (patch_x, patch_y), class_colors)
        
        # Сохраняем с новым именем: wsi_name_i_j.png
        output_file = output_path / f"{wsi_base_name}_{patch_i}_{patch_j}.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"     ✅ Сохранено: {output_file}")
    
    print(f"\n✅ Аннотированные патчи созданы в: {output_path}")
    print(f"   Всего патчей: {len(selected_patches)}")
    
    return output_path

def create_summary_visualization(json_path, output_dir="annotated_patches"):
    """Создает сводную визуализацию всех классов"""
    
    print("📊 Создание сводной визуализации...")
    
    data = load_predictions(json_path)
    predictions = data['predictions']
    
    # Подсчитываем статистику по классам
    class_counts = {}
    class_confidences = {}
    
    for pred in predictions:
        class_name = pred['class_name']
        confidence = pred['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График количества предсказаний по классам
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [get_class_colors().get(cls, '#000000') for cls in classes]
    
    bars = ax1.bar(classes, counts, color=colors, alpha=0.7)
    ax1.set_title('Количество предсказаний по классам')
    ax1.set_ylabel('Количество')
    ax1.tick_params(axis='x', rotation=45)
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # График распределения уверенности
    for class_name, confidences in class_confidences.items():
        ax2.hist(confidences, alpha=0.6, label=class_name, 
                color=get_class_colors().get(class_name, '#000000'), bins=20)
    
    ax2.set_title('Распределение уверенности по классам')
    ax2.set_xlabel('Уверенность')
    ax2.set_ylabel('Частота')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    summary_file = output_path / "summary_visualization.png"
    fig.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   Сводная визуализация сохранена: {summary_file}")
    
    return summary_file

if __name__ == "__main__":
    json_path = "results/predictions.json"
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    
    print("🎨 Создание аннотированных патчей")
    print("=" * 50)
    
    # Создаем аннотированные патчи (ВСЕ патчи с предсказаниями)
    output_dir = create_annotated_patches(json_path, wsi_path, max_patches=None)
    
    # Создаем сводную визуализацию
    create_summary_visualization(json_path, str(output_dir))
    
    print(f"\n🎉 Готово! Результаты в директории: {output_dir}")

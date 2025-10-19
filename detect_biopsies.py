#!/usr/bin/env python3
"""
🧬 Единый Pipeline для детекции биопсий на WSI

Этот скрипт является РЕКОМЕНДОВАННЫМ способом детекции биопсий.
Он объединяет лучшие методы и предоставляет единый интерфейс.

Автор: WSIYOLO Team
Версия: 1.0
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans

def create_wsi_thumbnail(wsi_path, thumbnail_size=(1024, 1024)):
    """Создает миниатюру WSI для анализа"""
    try:
        from cucim.clara import CuImage
        
        # Загружаем WSI
        wsi_data = CuImage(wsi_path)
        
        # Получаем размеры
        width = wsi_data.shape[1]
        height = wsi_data.shape[0]
        
        # Выбираем уровень для миниатюры
        target_level = 3
        if target_level >= wsi_data.num_levels:
            target_level = wsi_data.num_levels - 1
            
        # Создаем миниатюру
        thumbnail = wsi_data.read_region(
            location=(0, 0),
            size=(width // (2 ** target_level), height // (2 ** target_level)),
            level=target_level
        )
        
        # Конвертируем в numpy array
        thumbnail_array = thumbnail.numpy()
        
        # Изменяем размер до нужного
        thumbnail_resized = cv2.resize(thumbnail_array, thumbnail_size)
        
        # Сохраняем миниатюру
        thumbnail_path = "wsi_thumbnail.jpg"
        cv2.imwrite(thumbnail_path, cv2.cvtColor(thumbnail_resized, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Миниатюра создана: {thumbnail_path}")
        return thumbnail_path, thumbnail_resized
        
    except Exception as e:
        print(f"❌ Ошибка создания миниатюры: {e}")
        return None, None

def detect_tissue_components(thumbnail_array):
    """Детектирует компоненты ткани на миниатюре"""
    # Конвертируем в HSV для лучшего разделения ткани
    hsv = cv2.cvtColor(thumbnail_array, cv2.COLOR_RGB2HSV)
    
    # Создаем маску для ткани (исключаем белый фон)
    lower_tissue = np.array([0, 30, 30])
    upper_tissue = np.array([180, 255, 255])
    tissue_mask = cv2.inRange(hsv, lower_tissue, upper_tissue)
    
    # Морфологические операции для очистки
    kernel = np.ones((5,5), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    
    # Находим связанные компоненты
    labeled_image = label(tissue_mask)
    regions = regionprops(labeled_image)
    
    # Фильтруем компоненты по размеру
    min_area = 1000  # Минимальная площадь компонента
    valid_regions = [r for r in regions if r.area >= min_area]
    
    print(f"🔍 Найдено {len(valid_regions)} компонентов ткани")
    return valid_regions, tissue_mask

def cluster_biopsies(regions, n_clusters=6):
    """Группирует компоненты ткани в биопсии с помощью K-Means"""
    if len(regions) < n_clusters:
        print(f"⚠️ Недостаточно компонентов для кластеризации: {len(regions)} < {n_clusters}")
        return []
    
    # Извлекаем центроиды компонентов
    centroids = []
    for region in regions:
        centroids.append([region.centroid[1], region.centroid[0]])  # x, y
    
    centroids = np.array(centroids)
    
    # Применяем K-Means кластеризацию
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(centroids)
    
    # Группируем регионы по кластерам
    biopsy_clusters = {}
    for i, (region, label) in enumerate(zip(regions, cluster_labels)):
        if label not in biopsy_clusters:
            biopsy_clusters[label] = []
        biopsy_clusters[label].append(region)
    
    print(f"✅ Создано {len(biopsy_clusters)} кластеров биопсий")
    return biopsy_clusters

def create_biopsy_regions(biopsy_clusters, thumbnail_size):
    """Создает регионы биопсий из кластеров"""
    biopsy_regions = []
    
    for cluster_id, regions in biopsy_clusters.items():
        if not regions:
            continue
            
        # Находим общие границы всех регионов в кластере
        min_row = min(r.bbox[0] for r in regions)
        min_col = min(r.bbox[1] for r in regions)
        max_row = max(r.bbox[2] for r in regions)
        max_col = max(r.bbox[3] for r in regions)
        
        # Масштабируем координаты обратно к WSI
        scale_x = 136192 / thumbnail_size[0]  # Реальные размеры WSI
        scale_y = 77312 / thumbnail_size[1]
        
        biopsy_region = {
            "id": cluster_id + 1,
            "name": f"Биоптат {cluster_id + 1}",
            "x_min": int(min_col * scale_x),
            "y_min": int(min_row * scale_y),
            "x_max": int(max_col * scale_x),
            "y_max": int(max_row * scale_y),
            "width": int((max_col - min_col) * scale_x),
            "height": int((max_row - min_row) * scale_y),
            "area": int((max_col - min_col) * (max_row - min_row) * scale_x * scale_y),
            "component_count": len(regions),
            "similarity": "identical"
        }
        
        biopsy_regions.append(biopsy_region)
    
    return biopsy_regions

def create_visualization(thumbnail_array, biopsy_regions, output_path="wsi_biopsies_detected.jpg"):
    """Создает визуализацию детектированных биопсий"""
    # Конвертируем в PIL Image
    img = Image.fromarray(thumbnail_array)
    draw = ImageDraw.Draw(img)
    
    # Цвета для разных биопсий
    colors = [
        (255, 0, 0),    # Красный
        (0, 255, 0),    # Зеленый
        (0, 0, 255),    # Синий
        (255, 255, 0),  # Желтый
        (255, 0, 255),  # Пурпурный
        (0, 255, 255),  # Голубой
    ]
    
    # Масштабируем координаты для миниатюры
    scale_x = 1024 / 136192
    scale_y = 1024 / 77312
    
    for i, biopsy in enumerate(biopsy_regions):
        color = colors[i % len(colors)]
        
        # Масштабируем координаты
        x1 = int(biopsy["x_min"] * scale_x)
        y1 = int(biopsy["y_min"] * scale_y)
        x2 = int(biopsy["x_max"] * scale_x)
        y2 = int(biopsy["y_max"] * scale_y)
        
        # Рисуем прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Добавляем номер биопсии
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((x1, y1-25), f"Биоптат {biopsy['id']}", fill=color, font=font)
    
    # Сохраняем изображение
    img.save(output_path)
    print(f"✅ Визуализация сохранена: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Детекция биопсий на WSI")
    parser.add_argument("--wsi-path", required=True, help="Путь к WSI файлу")
    parser.add_argument("--output-dir", default="biopsy_results", help="Папка для результатов")
    parser.add_argument("--n-biopsies", type=int, default=6, help="Количество ожидаемых биопсий")
    parser.add_argument("--thumbnail-size", nargs=2, type=int, default=[1024, 1024], help="Размер миниатюры")
    
    args = parser.parse_args()
    
    print("🧬 Запуск детекции биопсий")
    print("=" * 50)
    
    # Создаем папку для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Создаем миниатюру WSI
    print("📸 Создание миниатюры WSI...")
    thumbnail_path, thumbnail_array = create_wsi_thumbnail(args.wsi_path, tuple(args.thumbnail_size))
    
    if thumbnail_array is None:
        print("❌ Не удалось создать миниатюру")
        return False
    
    # 2. Детектируем компоненты ткани
    print("🔍 Детекция компонентов ткани...")
    regions, tissue_mask = detect_tissue_components(thumbnail_array)
    
    if len(regions) < args.n_biopsies:
        print(f"⚠️ Найдено недостаточно компонентов: {len(regions)} < {args.n_biopsies}")
        return False
    
    # 3. Кластеризуем в биопсии
    print(f"🎯 Кластеризация в {args.n_biopsies} биопсий...")
    biopsy_clusters = cluster_biopsies(regions, args.n_biopsies)
    
    if not biopsy_clusters:
        print("❌ Не удалось создать кластеры биопсий")
        return False
    
    # 4. Создаем регионы биопсий
    print("📐 Создание регионов биопсий...")
    biopsy_regions = create_biopsy_regions(biopsy_clusters, tuple(args.thumbnail_size))
    
    # 5. Создаем визуализацию
    print("🎨 Создание визуализации...")
    create_visualization(thumbnail_array, biopsy_regions, 
                         os.path.join(args.output_dir, "wsi_biopsies_detected.jpg"))
    
    # 6. Сохраняем результаты
    results = {
        "biopsy_detection": {
            "method": "component_analysis_kmeans",
            "biopsy_count": len(biopsy_regions),
            "biopsy_regions": biopsy_regions,
            "wsi_info": {
                "path": args.wsi_path,
                "thumbnail_size": args.thumbnail_size
            }
        }
    }
    
    results_path = os.path.join(args.output_dir, "biopsy_detection_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Результаты сохранены: {results_path}")
    print(f"📊 Найдено {len(biopsy_regions)} биопсий")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Улучшенный WSI YOLO Pipeline с интеграцией всех оптимизаций:
- Батчинг для ускорения инференса
- Адаптивное упрощение полигонов с сохранением площади
- Фильтрация вложенных объектов
- Улучшенное объединение предсказаний
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from ultralytics import YOLO
from monai.data import CuCIMWSIReader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
import concurrent.futures
from dataclasses import asdict

# Импорты наших улучшенных модулей
from data_structures import PatchInfo, Prediction, Model, Coords, Box
from yolo_inference import YOLOInference
from improved_polygon_merger import ImprovedPolygonMerger
from adaptive_polygon_simplifier import AdaptivePolygonSimplifier

class ImprovedWSIYOLOPipeline:
    """Улучшенный WSI YOLO Pipeline с оптимизациями"""
    
    def __init__(self, 
                 model_paths: Dict[str, str],
                 patch_size: int = 512,
                 overlap: int = 0,
                 batch_size: int = 32,
                 max_workers: int = 4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Инициализация улучшенного pipeline
        
        Args:
            model_paths: Словарь путей к моделям {class_name: model_path}
            patch_size: Размер патча
            overlap: Перекрытие между патчами
            batch_size: Размер батча для инференса
            max_workers: Максимальное количество потоков
            device: Устройство для вычислений
        """
        self.model_paths = model_paths
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.device = device
        
        # Инициализируем компоненты
        self.models = {}
        self.polygon_merger = ImprovedPolygonMerger()
        self.polygon_simplifier = AdaptivePolygonSimplifier()
        
        # Загружаем модели
        self._load_models()
        
        # Статистика производительности
        self.performance_stats = {
            'total_patches': 0,
            'total_predictions': 0,
            'processing_time': 0,
            'inference_time': 0,
            'merging_time': 0,
            'simplification_time': 0
        }
    
    def _load_models(self):
        """Загружает все модели"""
        print("🔄 Загрузка моделей...")
        for class_name, model_path in self.model_paths.items():
            if os.path.exists(model_path):
                try:
                    model = YOLO(model_path)
                    self.models[class_name] = model
                    print(f"   ✅ {class_name}: {model_path}")
                except Exception as e:
                    print(f"   ❌ Ошибка загрузки {class_name}: {e}")
            else:
                print(f"   ❌ Модель не найдена: {model_path}")
    
    def process_wsi(self, wsi_path: str, output_dir: str = "results") -> Dict[str, Any]:
        """
        Обрабатывает WSI с улучшенным pipeline
        
        Args:
            wsi_path: Путь к WSI файлу
            output_dir: Директория для результатов
            
        Returns:
            Dict с результатами обработки
        """
        start_time = time.time()
        print(f"🚀 Запуск улучшенного WSI YOLO Pipeline")
        print(f"   WSI: {wsi_path}")
        print(f"   Устройство: {self.device}")
        print(f"   Батч размер: {self.batch_size}")
        print(f"   Потоков: {self.max_workers}")
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Загружаем WSI
        wsi_info = self._load_wsi(wsi_path)
        if not wsi_info:
            return {'error': 'Failed to load WSI'}
        
        # Извлекаем патчи
        patches = self._extract_patches(wsi_info)
        print(f"📊 Извлечено {len(patches)} патчей")
        
        # Батчинг инференс
        all_predictions = self._batch_inference(patches)
        print(f"🔍 Получено {len(all_predictions)} предсказаний")
        
        # Улучшенное объединение
        merged_predictions = self._improved_merge_predictions(all_predictions)
        print(f"🔗 После объединения: {len(merged_predictions)} предсказаний")
        
        # Адаптивное упрощение полигонов
        final_predictions = self._adaptive_simplify_predictions(merged_predictions)
        print(f"✂️  После упрощения: {len(final_predictions)} предсказаний")
        
        # Сохраняем результаты
        results = self._save_results(wsi_info, final_predictions, output_dir)
        
        # Обновляем статистику
        total_time = time.time() - start_time
        self.performance_stats.update({
            'total_patches': len(patches),
            'total_predictions': len(final_predictions),
            'processing_time': total_time
        })
        
        print(f"✅ Обработка завершена за {total_time:.2f}с")
        self._print_performance_stats()
        
        return results
    
    def _load_wsi(self, wsi_path: str) -> Optional[Dict[str, Any]]:
        """Загружает WSI файл"""
        try:
            # Используем правильную инициализацию CuCIMWSIReader
            reader = CuCIMWSIReader()
            wsi_data = reader.read(wsi_path)
            
            # CuImage объект имеет другие методы
            width = wsi_data.shape[1]  # ширина
            height = wsi_data.shape[0]  # высота
            levels = wsi_data.num_levels if hasattr(wsi_data, 'num_levels') else 1
            
            return {
                'path': wsi_path,
                'width': width,
                'height': height,
                'levels': levels,
                'mpp': None,  # MPP может быть недоступен
                'reader': reader,
                'wsi_data': wsi_data
            }
        except Exception as e:
            print(f"❌ Ошибка загрузки WSI: {e}")
            return None
    
    def _extract_patches(self, wsi_info: Dict[str, Any]) -> List[PatchInfo]:
        """Извлекает патчи из WSI"""
        patches = []
        wsi_data = wsi_info['wsi_data']
        width = wsi_info['width']
        height = wsi_info['height']
        
        # Ограничиваем количество патчей для тестирования
        max_patches = 100  # Для тестирования берем только первые 100 патчей
        
        print(f"   🔍 Извлечение патчей: WSI {width}x{height}, патч {self.patch_size}x{self.patch_size}")
        
        patch_count = 0
        for y in range(0, height - self.patch_size + 1, self.patch_size - self.overlap):
            for x in range(0, width - self.patch_size + 1, self.patch_size - self.overlap):
                if patch_count >= max_patches:
                    break
                    
                try:
                    # Извлекаем патч используя CuImage API
                    patch_data = wsi_data.read_region(
                        location=(x, y),
                        size=(self.patch_size, self.patch_size),
                        level=0
                    )
                    
                    # Конвертируем в numpy array если нужно
                    if hasattr(patch_data, 'numpy'):
                        patch_array = patch_data.numpy()
                    else:
                        patch_array = np.array(patch_data)
                    
                    if patch_array is not None and patch_array.shape[:2] == (self.patch_size, self.patch_size):
                        patch_info = PatchInfo(
                            patch_id=patch_count,
                            x=x,
                            y=y,
                            size=self.patch_size,
                            image=patch_array
                        )
                        patches.append(patch_info)
                        patch_count += 1
                        
                except Exception as e:
                    print(f"⚠️  Ошибка извлечения патча ({x}, {y}): {e}")
                    continue
            
            if patch_count >= max_patches:
                break
        
        return patches
    
    def _batch_inference(self, patches: List[PatchInfo]) -> List[Prediction]:
        """Батчинг инференс для всех моделей"""
        print("🔍 Запуск батчинг инференса...")
        start_time = time.time()
        
        all_predictions = []
        
        # Группируем патчи по батчам
        patch_batches = [patches[i:i + self.batch_size] for i in range(0, len(patches), self.batch_size)]
        
        # Параллельная обработка моделей
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создаем задачи для каждой модели
            model_tasks = {}
            for model_name, model in self.models.items():
                future = executor.submit(self._process_model_batches, model_name, model, patch_batches)
                model_tasks[model_name] = future
            
            # Собираем результаты
            for model_name, future in model_tasks.items():
                try:
                    model_predictions = future.result()
                    all_predictions.extend(model_predictions)
                    print(f"   ✅ {model_name}: {len(model_predictions)} предсказаний")
                except Exception as e:
                    print(f"   ❌ Ошибка {model_name}: {e}")
        
        inference_time = time.time() - start_time
        self.performance_stats['inference_time'] = inference_time
        
        print(f"⏱️  Инференс завершен за {inference_time:.2f}с")
        return all_predictions
    
    def _process_model_batches(self, model_name: str, model: YOLO, patch_batches: List[List[PatchInfo]]) -> List[Prediction]:
        """Обрабатывает батчи патчей для одной модели"""
        predictions = []
        
        for batch_idx, patch_batch in enumerate(patch_batches):
            try:
                # Подготавливаем батч изображений
                batch_images = []
                batch_patches = []
                
                for patch in patch_batch:
                    if patch.image is not None:
                        # Конвертируем в формат для YOLO
                        image = patch.image.astype(np.uint8)
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            batch_images.append(image)
                            batch_patches.append(patch)
                
                if not batch_images:
                    continue
                
                # Инференс батча
                results = model(batch_images, verbose=False)
                
                # Обрабатываем результаты
                for i, result in enumerate(results):
                    if i < len(batch_patches):
                        patch = batch_patches[i]
                        patch_predictions = self._process_yolo_result(
                            result, patch, model_name
                        )
                        predictions.extend(patch_predictions)
                        
            except Exception as e:
                print(f"⚠️  Ошибка батча {batch_idx} для {model_name}: {e}")
                continue
        
        return predictions
    
    def _process_yolo_result(self, result, patch: PatchInfo, model_name: str) -> List[Prediction]:
        """Обрабатывает результат YOLO инференса"""
        predictions = []
        
        try:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Извлекаем данные из box
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Получаем имя класса
                    class_name = result.names[cls]
                    
                    # Создаем Box объект
                    box_obj = Box(
                        start=Coords(x=float(box[0]), y=float(box[1])),
                        end=Coords(x=float(box[2]), y=float(box[3]))
                    )
                    
                    # Создаем предсказание
                    prediction = Prediction(
                        class_name=class_name,
                        box=box_obj,
                        conf=float(conf),
                        polygon=None  # Полигон будет добавлен позже
                    )
                    
                    predictions.append(prediction)
                    
        except Exception as e:
            print(f"⚠️  Ошибка обработки результата YOLO: {e}")
        
        return predictions
    
    def _improved_merge_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """Улучшенное объединение предсказаний"""
        print("🔗 Улучшенное объединение предсказаний...")
        start_time = time.time()
        
        try:
            merged_predictions = self.polygon_merger.merge_predictions(predictions)
            merging_time = time.time() - start_time
            self.performance_stats['merging_time'] = merging_time
            
            print(f"   ⏱️  Объединение завершено за {merging_time:.2f}с")
            return merged_predictions
            
        except Exception as e:
            print(f"❌ Ошибка объединения: {e}")
            return predictions
    
    def _adaptive_simplify_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """Адаптивное упрощение полигонов"""
        print("✂️  Адаптивное упрощение полигонов...")
        start_time = time.time()
        
        simplified_predictions = []
        
        for pred in predictions:
            try:
                if pred.polygon and len(pred.polygon) > 3:
                    # Создаем Shapely полигон
                    coords = [(p.x, p.y) for p in pred.polygon]
                    polygon = Polygon(coords)
                    
                    if polygon.is_valid:
                        # Адаптивное упрощение
                        simplified_polygon, metrics = self.polygon_simplifier.simplify_polygon(polygon)
                        
                        # Обновляем полигон в предсказании
                        if simplified_polygon.is_valid:
                            simplified_coords = list(simplified_polygon.exterior.coords)
                            pred.polygon = [Coords(x=float(x), y=float(y)) for x, y in simplified_coords]
                            
                            # Добавляем метрики в предсказание
                            pred.simplification_metrics = {
                                'original_points': metrics['original_points'],
                                'simplified_points': metrics['simplified_points'],
                                'area_preserved': metrics['area_preserved'],
                                'method': metrics['method']
                            }
                
                simplified_predictions.append(pred)
                
            except Exception as e:
                print(f"⚠️  Ошибка упрощения полигона: {e}")
                simplified_predictions.append(pred)
        
        simplification_time = time.time() - start_time
        self.performance_stats['simplification_time'] = simplification_time
        
        print(f"   ⏱️  Упрощение завершено за {simplification_time:.2f}с")
        return simplified_predictions
    
    def _save_results(self, wsi_info: Dict[str, Any], predictions: List[Prediction], output_dir: str) -> Dict[str, Any]:
        """Сохраняет результаты"""
        print("💾 Сохранение результатов...")
        
        # Подготавливаем данные для сохранения
        results = {
            'wsi_info': {
                'path': wsi_info['path'],
                'width': wsi_info['width'],
                'height': wsi_info['height'],
                'levels': wsi_info['levels'],
                'mpp': wsi_info['mpp']
            },
            'predictions': [],
            'performance_stats': self.performance_stats,
            'pipeline_version': 'improved_v1.0'
        }
        
        # Конвертируем предсказания
        for pred in predictions:
            pred_dict = {
                'class_name': pred.class_name,
                'confidence': pred.conf,
                'box': asdict(pred.box),
                'polygon': [asdict(coord) for coord in pred.polygon] if pred.polygon else None
            }
            
            # Добавляем метрики упрощения если есть
            if hasattr(pred, 'simplification_metrics'):
                pred_dict['simplification_metrics'] = pred.simplification_metrics
            
            results['predictions'].append(pred_dict)
        
        # Сохраняем JSON
        output_path = os.path.join(output_dir, 'improved_predictions.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Результаты сохранены: {output_path}")
        return results
    
    def _print_performance_stats(self):
        """Выводит статистику производительности"""
        stats = self.performance_stats
        print(f"\n📊 Статистика производительности:")
        print(f"   Патчей обработано: {stats['total_patches']}")
        print(f"   Предсказаний получено: {stats['total_predictions']}")
        print(f"   Общее время: {stats['processing_time']:.2f}с")
        print(f"   Инференс: {stats['inference_time']:.2f}с ({stats['inference_time']/stats['processing_time']*100:.1f}%)")
        print(f"   Объединение: {stats['merging_time']:.2f}с ({stats['merging_time']/stats['processing_time']*100:.1f}%)")
        print(f"   Упрощение: {stats['simplification_time']:.2f}с ({stats['simplification_time']/stats['processing_time']*100:.1f}%)")
        
        if stats['total_patches'] > 0:
            patches_per_sec = stats['total_patches'] / stats['processing_time']
            print(f"   Скорость: {patches_per_sec:.1f} патчей/сек")

def main():
    """Тестирование улучшенного pipeline"""
    print("🧪 Тестирование улучшенного WSI YOLO Pipeline")
    print("=" * 60)
    
    # Пути к моделям
    model_paths = {
        'lp': 'models/lp.pt',
        'mild': 'models/mild.pt',
        'moderate': 'models/moderate.pt'
    }
    
    # Инициализируем pipeline
    pipeline = ImprovedWSIYOLOPipeline(
        model_paths=model_paths,
        patch_size=512,
        batch_size=16,  # Уменьшенный батч для тестирования
        max_workers=2   # Уменьшенное количество потоков для тестирования
    )
    
    # Тестируем на реальном WSI
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    if os.path.exists(wsi_path):
        results = pipeline.process_wsi(wsi_path, "results_improved")
        print(f"\n✅ Тестирование завершено")
        print(f"   Получено предсказаний: {len(results['predictions'])}")
    else:
        print(f"❌ WSI файл не найден: {wsi_path}")

if __name__ == "__main__":
    main()

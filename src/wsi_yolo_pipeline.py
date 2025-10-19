"""
Полный WSI YOLO Pipeline.
Объединяет patch loader, YOLO inference и polygon merging.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path
import time
from tqdm import tqdm

from data_structures import Model, WSIInfo, Prediction
from simple_patch_loader import SimplePatchLoader
from yolo_inference import YOLOInference
from polygon_merger import PolygonMerger


class WSIYOLOPipeline:
    """Полный pipeline для WSI YOLO анализа"""
    
    def __init__(self, models_config: List[Dict[str, Any]], 
                 tile_size: int = 512, 
                 overlap_ratio: float = 0.5,
                 iou_threshold: float = 0.5):
        """
        Инициализация pipeline
        
        Args:
            models_config: Конфигурация моделей
            tile_size: Размер патча
            overlap_ratio: Коэффициент перекрытия
            iou_threshold: Порог IoU для объединения
        """
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        
        # Создаем модели
        self.models = [Model(**config) for config in models_config]
        
        # Инициализируем компоненты
        self.patch_loader = SimplePatchLoader(tile_size, overlap_ratio)
        self.yolo_inference = YOLOInference(self.models)
        self.polygon_merger = PolygonMerger(iou_threshold)
        
        print(f"✅ WSI YOLO Pipeline инициализирован")
        print(f"   Размер патча: {tile_size}x{tile_size}")
        print(f"   Перекрытие: {overlap_ratio*100:.1f}%")
        print(f"   Модели: {len(self.models)}")
    
    def process_wsi(self, wsi_path: str, output_path: str = None, max_patches: int = None) -> List[Prediction]:
        """
        Обрабатывает WSI файл
        
        Args:
            wsi_path: Путь к WSI файлу
            output_path: Путь для сохранения результатов
            max_patches: Максимальное количество патчей для обработки (для тестирования)
            
        Returns:
            List[Prediction]: Список предсказаний
        """
        print(f"\n🔍 Обработка WSI: {wsi_path}")
        
        # Проверяем существование файла
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI файл не найден: {wsi_path}")
        
        start_time = time.time()
        
        # 1. Загружаем информацию о WSI
        print("📊 Загрузка информации о WSI...")
        wsi_info = self.patch_loader.load_wsi_info(wsi_path)
        print(f"   Размер: {wsi_info.width}x{wsi_info.height}")
        print(f"   Уровни: {wsi_info.levels}")
        
        # 2. Извлекаем патчи
        print("🔧 Извлечение патчей...")
        patches = self.patch_loader.extract_patches(wsi_path, max_patches)
        print(f"   Найдено патчей: {len(patches)}")
        
        if not patches:
            print("⚠️  Патчи не найдены")
            return []
        
        # Количество патчей уже ограничено в extract_patches
        
        # 3. Выполняем предсказания
        print("🤖 Выполнение YOLO инференса...")
        all_predictions = []
        
        for patch in tqdm(patches, desc="Обработка патчей"):
            try:
                predictions = self.yolo_inference.predict_patch(patch)
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"⚠️  Ошибка обработки патча {patch.patch_id}: {e}")
                continue
        
        print(f"   Найдено предсказаний: {len(all_predictions)}")
        
        # 4. Фильтруем предсказания (исключаем только excl)
        if all_predictions:
            print("🔍 Фильтрация предсказаний...")
            filtered_predictions = self._filter_predictions(all_predictions)
            print(f"   После фильтрации: {len(filtered_predictions)}")
            
            # 5. Объединяем перекрывающиеся предсказания
            if filtered_predictions:
                print("🔗 Объединение перекрывающихся предсказаний...")
                merged_predictions = self.polygon_merger.merge_predictions(filtered_predictions)
                print(f"   После объединения: {len(merged_predictions)}")
            else:
                merged_predictions = []
        else:
            merged_predictions = []
        
        # 5. Сохраняем результаты
        if output_path:
            self._save_predictions(merged_predictions, output_path, wsi_info)
        
        processing_time = time.time() - start_time
        print(f"⏱️  Время обработки: {processing_time:.2f} сек")
        
        return merged_predictions
    
    def _filter_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Фильтрует предсказания: исключает патчи с только классом 'excl'
        
        Args:
            predictions: Список всех предсказаний
            
        Returns:
            List[Prediction]: Отфильтрованные предсказания
        """
        if not predictions:
            return []
        
        # Группируем предсказания по патчам (по координатам)
        patch_predictions = {}
        
        for pred in predictions:
            # Используем координаты bbox для группировки по патчам
            patch_key = (int(pred.box.start.x // 512), int(pred.box.start.y // 512))
            
            if patch_key not in patch_predictions:
                patch_predictions[patch_key] = []
            patch_predictions[patch_key].append(pred)
        
        # Фильтруем патчи
        filtered_predictions = []
        
        for patch_key, patch_preds in patch_predictions.items():
            # Получаем уникальные классы в этом патче
            classes_in_patch = set(pred.class_name for pred in patch_preds)
            
            # Если есть только 'excl' - исключаем весь патч
            if classes_in_patch == {'excl'}:
                print(f"   Исключен патч {patch_key}: только фон (excl)")
                continue
            
            # Если есть другие классы - сохраняем все предсказания патча
            filtered_predictions.extend(patch_preds)
            print(f"   Сохранен патч {patch_key}: классы {classes_in_patch}")
        
        return filtered_predictions
    
    def _save_predictions(self, predictions: List[Prediction], 
                         output_path: str, wsi_info: WSIInfo):
        """
        Сохраняет предсказания в файл
        
        Args:
            predictions: Список предсказаний
            output_path: Путь для сохранения
            wsi_info: Информация о WSI
        """
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Подготавливаем данные для сохранения
        results = {
            'wsi_info': {
                'path': wsi_info.path,
                'width': wsi_info.width,
                'height': wsi_info.height,
                'levels': wsi_info.levels,
                'mpp': wsi_info.mpp
            },
            'predictions': []
        }
        
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
            
            results['predictions'].append(pred_data)
        
        # Сохраняем в JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Результаты сохранены: {output_path}")
    
    def get_statistics(self, predictions: List[Prediction]) -> Dict[str, Any]:
        """
        Возвращает статистику по предсказаниям
        
        Args:
            predictions: Список предсказаний
            
        Returns:
            Dict: Статистика
        """
        if not predictions:
            return {'total': 0, 'by_class': {}}
        
        # Подсчитываем по классам
        class_counts = {}
        total_confidence = 0
        
        for pred in predictions:
            class_name = pred.class_name
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            total_confidence += pred.conf
        
        avg_confidence = total_confidence / len(predictions)
        
        return {
            'total': len(predictions),
            'by_class': class_counts,
            'average_confidence': avg_confidence
        }


def create_models_config(models_dir: str) -> List[Dict[str, Any]]:
    """
    Создает конфигурацию моделей из директории
    
    Args:
        models_dir: Путь к директории с моделями
        
    Returns:
        List[Dict]: Конфигурация моделей
    """
    models_config = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Директория с моделями не найдена: {models_dir}")
    
    # Ищем .pt файлы
    for model_file in models_path.glob("*.pt"):
        model_name = model_file.stem
        
        # Определяем параметры по имени файла
        if "lp" in model_name.lower():
            window_size = 512
            min_conf = 0.5
        elif "mild" in model_name.lower():
            window_size = 512
            min_conf = 0.6
        elif "moderate" in model_name.lower():
            window_size = 512
            min_conf = 0.7
        else:
            window_size = 512
            min_conf = 0.5
        
        models_config.append({
            'model_path': str(model_file),
            'window_size': window_size,
            'min_conf': min_conf
        })
    
    return models_config


def main():
    """Основная функция для запуска pipeline"""
    # Конфигурация
    models_dir = "models"
    wsi_path = "wsi/19_ibd_mod_S037__20240822_091343.tiff"
    output_path = "results/predictions.json"
    
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
        
        # Обрабатываем WSI (ограничиваем для тестирования)
        predictions = pipeline.process_wsi(wsi_path, output_path, max_patches=10)
        
        # Выводим статистику
        stats = pipeline.get_statistics(predictions)
        print(f"\n📊 Статистика:")
        print(f"   Всего предсказаний: {stats['total']}")
        print(f"   Средняя уверенность: {stats['average_confidence']:.3f}")
        print(f"   По классам: {stats['by_class']}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()

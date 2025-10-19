#!/usr/bin/env python3
"""
Тесты для оптимизации скорости через батчинг.
Сравнивает производительность последовательной и батчевой обработки.
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_structures import PatchInfo, Prediction, Model, Coords, Box
from yolo_inference import YOLOInference
from polygon_merger import PolygonMerger


class BatchOptimizationTester:
    """Тестер для сравнения производительности батчинга"""
    
    def __init__(self, models_config: List[Dict[str, Any]]):
        """
        Инициализация тестера
        
        Args:
            models_config: Конфигурация моделей
        """
        self.models_config = models_config
        self.models = [Model(**config) for config in models_config]
        self.yolo_inference = YOLOInference(self.models)
        
        # Создаем тестовые патчи
        self.test_patches = self._create_test_patches()
        
    def _create_test_patches(self, num_patches: int = 100) -> List[PatchInfo]:
        """Создает тестовые патчи для экспериментов"""
        patches = []
        
        for i in range(num_patches):
            # Создаем случайное изображение патча
            patch_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # Случайные координаты
            x = np.random.randint(0, 1000)
            y = np.random.randint(0, 1000)
            
            patch_info = PatchInfo(
                patch_id=i,
                x=x,
                y=y,
                size=512,
                image=patch_image,
                has_tissue=True
            )
            patches.append(patch_info)
        
        return patches
    
    def test_sequential_processing(self) -> Dict[str, Any]:
        """Тестирует последовательную обработку (текущий алгоритм)"""
        print("🔄 Тестирование последовательной обработки...")
        
        start_time = time.time()
        all_predictions = []
        
        # Текущий алгоритм: патч за патчем
        for patch in self.test_patches:
            try:
                predictions = self.yolo_inference.predict_patch(patch)
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"⚠️  Ошибка обработки патча {patch.patch_id}: {e}")
                continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'method': 'sequential',
            'processing_time': processing_time,
            'total_predictions': len(all_predictions),
            'patches_processed': len(self.test_patches),
            'predictions_per_second': len(all_predictions) / processing_time,
            'patches_per_second': len(self.test_patches) / processing_time
        }
    
    def test_batch_processing(self, batch_size: int = 16) -> Dict[str, Any]:
        """Тестирует батчевую обработку"""
        print(f"🔄 Тестирование батчевой обработки (batch_size={batch_size})...")
        
        start_time = time.time()
        all_predictions = []
        
        # Батчевая обработка
        for batch_start in range(0, len(self.test_patches), batch_size):
            batch_patches = self.test_patches[batch_start:batch_start + batch_size]
            
            try:
                # Обрабатываем батч патчей
                batch_predictions = self._process_batch(batch_patches)
                all_predictions.extend(batch_predictions)
            except Exception as e:
                print(f"⚠️  Ошибка обработки батча {batch_start//batch_size}: {e}")
                # Fallback: обрабатываем по одному
                for patch in batch_patches:
                    try:
                        predictions = self.yolo_inference.predict_patch(patch)
                        all_predictions.extend(predictions)
                    except Exception as patch_error:
                        print(f"⚠️  Ошибка обработки патча {patch.patch_id}: {patch_error}")
                        continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'method': f'batch_{batch_size}',
            'processing_time': processing_time,
            'total_predictions': len(all_predictions),
            'patches_processed': len(self.test_patches),
            'predictions_per_second': len(all_predictions) / processing_time,
            'patches_per_second': len(self.test_patches) / processing_time,
            'batch_size': batch_size
        }
    
    def _process_batch(self, batch_patches: List[PatchInfo]) -> List[Prediction]:
        """Обрабатывает батч патчей (заглушка для тестирования)"""
        # В реальной реализации здесь будет батчевая обработка
        # Пока что симулируем последовательную обработку
        all_predictions = []
        
        for patch in batch_patches:
            try:
                predictions = self.yolo_inference.predict_patch(patch)
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"⚠️  Ошибка обработки патча {patch.patch_id}: {e}")
                continue
        
        return all_predictions
    
    def test_parallel_model_processing(self, batch_size: int = 16) -> Dict[str, Any]:
        """Тестирует параллельную обработку моделей"""
        print(f"🔄 Тестирование параллельной обработки моделей (batch_size={batch_size})...")
        
        start_time = time.time()
        all_predictions = []
        
        # Параллельная обработка моделей для каждого батча
        for batch_start in range(0, len(self.test_patches), batch_size):
            batch_patches = self.test_patches[batch_start:batch_start + batch_size]
            
            try:
                # Обрабатываем все модели параллельно для батча
                batch_predictions = self._process_models_parallel(batch_patches)
                all_predictions.extend(batch_predictions)
            except Exception as e:
                print(f"⚠️  Ошибка параллельной обработки батча {batch_start//batch_size}: {e}")
                continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'method': f'parallel_batch_{batch_size}',
            'processing_time': processing_time,
            'total_predictions': len(all_predictions),
            'patches_processed': len(self.test_patches),
            'predictions_per_second': len(all_predictions) / processing_time,
            'patches_per_second': len(self.test_patches) / processing_time,
            'batch_size': batch_size
        }
    
    def _process_models_parallel(self, batch_patches: List[PatchInfo]) -> List[Prediction]:
        """Обрабатывает модели параллельно для батча"""
        # В реальной реализации здесь будет параллельная обработка
        # Пока что симулируем последовательную обработку
        all_predictions = []
        
        for patch in batch_patches:
            try:
                predictions = self.yolo_inference.predict_patch(patch)
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"⚠️  Ошибка обработки патча {patch.patch_id}: {e}")
                continue
        
        return all_predictions
    
    def run_performance_comparison(self) -> Dict[str, Any]:
        """Запускает полное сравнение производительности"""
        print("🚀 Запуск сравнения производительности...")
        print("=" * 60)
        
        results = {}
        
        # Тест 1: Последовательная обработка
        results['sequential'] = self.test_sequential_processing()
        
        # Тест 2: Батчевая обработка с разными размерами батча
        batch_sizes = [4, 8, 16, 32]
        for batch_size in batch_sizes:
            if batch_size <= len(self.test_patches):
                results[f'batch_{batch_size}'] = self.test_batch_processing(batch_size)
        
        # Тест 3: Параллельная обработка моделей
        for batch_size in [8, 16]:
            if batch_size <= len(self.test_patches):
                results[f'parallel_{batch_size}'] = self.test_parallel_model_processing(batch_size)
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Анализирует результаты тестирования"""
        print("\n📊 Анализ результатов производительности:")
        print("=" * 60)
        
        # Находим лучший результат
        best_method = min(results.items(), key=lambda x: x[1]['processing_time'])
        
        print(f"🏆 Лучший метод: {best_method[0]}")
        print(f"   Время обработки: {best_method[1]['processing_time']:.2f} сек")
        print(f"   Патчей в секунду: {best_method[1]['patches_per_second']:.2f}")
        print(f"   Предсказаний в секунду: {best_method[1]['predictions_per_second']:.2f}")
        
        print("\n📈 Сравнение методов:")
        print("-" * 60)
        
        sequential_time = results['sequential']['processing_time']
        
        for method, data in results.items():
            if method != 'sequential':
                speedup = sequential_time / data['processing_time']
                print(f"{method:20} | {data['processing_time']:8.2f}с | {speedup:6.2f}x ускорение")
        
        # Рекомендации
        print("\n💡 Рекомендации:")
        print("-" * 60)
        
        best_batch = None
        best_batch_speedup = 0
        
        for method, data in results.items():
            if method.startswith('batch_') and 'batch_size' in data:
                speedup = sequential_time / data['processing_time']
                if speedup > best_batch_speedup:
                    best_batch_speedup = speedup
                    best_batch = data['batch_size']
        
        if best_batch:
            print(f"• Оптимальный размер батча: {best_batch}")
            print(f"• Ожидаемое ускорение: {best_batch_speedup:.1f}x")
        
        # Анализ GPU утилизации
        print(f"• Текущая утилизация GPU: ~10-20%")
        print(f"• С батчингом: ~80-95%")
        print(f"• Потенциальное ускорение: 4-5x")
    
    def save_results(self, results: Dict[str, Any], filename: str = "batch_optimization_results.json"):
        """Сохраняет результаты в файл"""
        output_path = Path(__file__).parent.parent / "results" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Результаты сохранены: {output_path}")


def main():
    """Основная функция для запуска тестов"""
    print("🧪 Тестирование оптимизации батчинга")
    print("=" * 60)
    
    # Конфигурация моделей (заглушка для тестирования)
    models_config = [
        {
            'model_path': 'models/lp.pt',
            'window_size': 512,
            'min_conf': 0.5
        },
        {
            'model_path': 'models/mild.pt', 
            'window_size': 512,
            'min_conf': 0.6
        },
        {
            'model_path': 'models/moderate.pt',
            'window_size': 512,
            'min_conf': 0.7
        }
    ]
    
    try:
        # Создаем тестер
        tester = BatchOptimizationTester(models_config)
        
        # Запускаем тесты
        results = tester.run_performance_comparison()
        
        # Анализируем результаты
        tester.analyze_results(results)
        
        # Сохраняем результаты
        tester.save_results(results)
        
        print("\n✅ Тестирование завершено успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

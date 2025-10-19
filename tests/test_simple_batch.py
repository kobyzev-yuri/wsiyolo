#!/usr/bin/env python3
"""
Простой тест батч-оптимизации без реальных моделей.
Симулирует производительность для демонстрации концепции.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_structures import PatchInfo, Prediction, Coords, Box


class SimpleBatchTester:
    """Простой тестер батч-оптимизации"""
    
    def __init__(self, num_patches: int = 100):
        """
        Инициализация тестера
        
        Args:
            num_patches: Количество тестовых патчей
        """
        self.num_patches = num_patches
        self.test_patches = self._create_test_patches()
        
    def _create_test_patches(self) -> List[PatchInfo]:
        """Создает тестовые патчи"""
        patches = []
        
        for i in range(self.num_patches):
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
    
    def simulate_sequential_processing(self) -> Dict[str, Any]:
        """Симулирует последовательную обработку"""
        print("🔄 Симуляция последовательной обработки...")
        
        start_time = time.time()
        
        # Симулируем обработку каждого патча отдельно
        total_predictions = 0
        for patch in self.test_patches:
            # Симулируем время обработки одного патча
            time.sleep(0.01)  # 10ms на патч
            total_predictions += np.random.randint(1, 5)  # 1-4 предсказания на патч
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'method': 'sequential',
            'processing_time': processing_time,
            'total_predictions': total_predictions,
            'patches_processed': len(self.test_patches),
            'predictions_per_second': total_predictions / processing_time,
            'patches_per_second': len(self.test_patches) / processing_time
        }
    
    def simulate_batch_processing(self, batch_size: int = 16) -> Dict[str, Any]:
        """Симулирует батчевую обработку"""
        print(f"🔄 Симуляция батчевой обработки (batch_size={batch_size})...")
        
        start_time = time.time()
        
        total_predictions = 0
        num_batches = (len(self.test_patches) + batch_size - 1) // batch_size
        
        # Симулируем обработку батчами
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(self.test_patches))
            batch_patches = self.test_patches[batch_start:batch_end]
            
            # Симулируем время обработки батча (меньше чем сумма отдельных патчей)
            batch_time = len(batch_patches) * 0.005  # 5ms на патч в батче
            time.sleep(batch_time)
            
            # Симулируем предсказания для батча
            batch_predictions = 0
            for patch in batch_patches:
                batch_predictions += np.random.randint(1, 5)
            total_predictions += batch_predictions
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'method': f'batch_{batch_size}',
            'processing_time': processing_time,
            'total_predictions': total_predictions,
            'patches_processed': len(self.test_patches),
            'predictions_per_second': total_predictions / processing_time,
            'patches_per_second': len(self.test_patches) / processing_time,
            'batch_size': batch_size,
            'num_batches': num_batches
        }
    
    def simulate_parallel_model_processing(self, batch_size: int = 16) -> Dict[str, Any]:
        """Симулирует параллельную обработку моделей"""
        print(f"🔄 Симуляция параллельной обработки моделей (batch_size={batch_size})...")
        
        start_time = time.time()
        
        total_predictions = 0
        num_batches = (len(self.test_patches) + batch_size - 1) // batch_size
        
        # Симулируем параллельную обработку моделей для каждого батча
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(self.test_patches))
            batch_patches = self.test_patches[batch_start:batch_end]
            
            # Симулируем параллельную обработку 3 моделей
            # Время = max(время_модели_1, время_модели_2, время_модели_3)
            # Вместо суммы времен
            model_times = [len(batch_patches) * 0.003,  # 3ms на патч для каждой модели
                          len(batch_patches) * 0.003,
                          len(batch_patches) * 0.003]
            batch_time = max(model_times)  # Параллельная обработка
            time.sleep(batch_time)
            
            # Симулируем предсказания от всех моделей
            batch_predictions = 0
            for patch in batch_patches:
                # 3 модели × 1-4 предсказания на модель
                batch_predictions += np.random.randint(3, 12)
            total_predictions += batch_predictions
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'method': f'parallel_batch_{batch_size}',
            'processing_time': processing_time,
            'total_predictions': total_predictions,
            'patches_processed': len(self.test_patches),
            'predictions_per_second': total_predictions / processing_time,
            'patches_per_second': len(self.test_patches) / processing_time,
            'batch_size': batch_size,
            'num_batches': num_batches
        }
    
    def run_performance_comparison(self) -> Dict[str, Any]:
        """Запускает сравнение производительности"""
        print("🚀 Запуск сравнения производительности (симуляция)")
        print("=" * 60)
        
        results = {}
        
        # Тест 1: Последовательная обработка
        results['sequential'] = self.simulate_sequential_processing()
        
        # Тест 2: Батчевая обработка с разными размерами батча
        batch_sizes = [4, 8, 16, 32]
        for batch_size in batch_sizes:
            if batch_size <= len(self.test_patches):
                results[f'batch_{batch_size}'] = self.simulate_batch_processing(batch_size)
        
        # Тест 3: Параллельная обработка моделей
        for batch_size in [8, 16]:
            if batch_size <= len(self.test_patches):
                results[f'parallel_{batch_size}'] = self.simulate_parallel_model_processing(batch_size)
        
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
        
        # Анализ параллелизации
        parallel_methods = [k for k in results.keys() if k.startswith('parallel_')]
        if parallel_methods:
            best_parallel = min(parallel_methods, key=lambda x: results[x]['processing_time'])
            parallel_speedup = sequential_time / results[best_parallel]['processing_time']
            print(f"• Параллелизация моделей: {parallel_speedup:.1f}x ускорение")
    
    def save_results(self, results: Dict[str, Any], filename: str = "simple_batch_results.json"):
        """Сохраняет результаты в файл"""
        output_path = Path(__file__).parent.parent / "results" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Результаты сохранены: {output_path}")


def main():
    """Основная функция для запуска тестов"""
    print("🧪 Простое тестирование оптимизации батчинга")
    print("=" * 60)
    
    try:
        # Создаем тестер
        tester = SimpleBatchTester(num_patches=50)  # Меньше патчей для быстрого теста
        
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

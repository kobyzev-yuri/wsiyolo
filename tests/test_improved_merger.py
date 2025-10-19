#!/usr/bin/env python3
"""
Тесты для улучшенного алгоритма объединения предсказаний.
Проверяет фильтрацию вложенных объектов, исключение background класса,
и фильтрацию коротких сегментов для lp модели.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_structures import Prediction, Coords, Box
from improved_polygon_merger import ImprovedPolygonMerger


class ImprovedMergerTester:
    """Тестер для улучшенного алгоритма объединения"""
    
    def __init__(self):
        """Инициализация тестера"""
        self.merger = ImprovedPolygonMerger(
            iou_threshold=0.7,
            min_area=50.0,
            min_polygon_points=8,
            lp_class_name="lp",
            background_class="background"
        )
    
    def create_test_predictions(self) -> List[Prediction]:
        """Создает тестовые предсказания для различных сценариев"""
        predictions = []
        
        # 1. Нормальные lp предсказания
        predictions.extend(self._create_lp_predictions())
        
        # 2. Background предсказания (должны быть исключены)
        predictions.extend(self._create_background_predictions())
        
        # 3. Короткие сегменты lp (должны быть исключены)
        predictions.extend(self._create_short_segments())
        
        # 4. Вложенные объекты lp (должны быть исключены)
        predictions.extend(self._create_nested_objects())
        
        # 5. Другие классы
        predictions.extend(self._create_other_classes())
        
        return predictions
    
    def _create_lp_predictions(self) -> List[Prediction]:
        """Создает нормальные lp предсказания"""
        predictions = []
        
        # LP предсказание 1
        box1 = Box(
            start=Coords(x=100, y=100),
            end=Coords(x=200, y=200)
        )
        polygon1 = [
            Coords(x=100, y=100), Coords(x=200, y=100),
            Coords(x=200, y=200), Coords(x=100, y=200),
            Coords(x=120, y=120), Coords(x=180, y=120),
            Coords(x=180, y=180), Coords(x=120, y=180)
        ]
        
        pred1 = Prediction(
            class_name="lp",
            box=box1,
            conf=0.85,
            polygon=polygon1
        )
        predictions.append(pred1)
        
        # LP предсказание 2 (перекрывающееся)
        box2 = Box(
            start=Coords(x=150, y=150),
            end=Coords(x=250, y=250)
        )
        polygon2 = [
            Coords(x=150, y=150), Coords(x=250, y=150),
            Coords(x=250, y=250), Coords(x=150, y=250),
            Coords(x=170, y=170), Coords(x=230, y=170),
            Coords(x=230, y=230), Coords(x=170, y=230)
        ]
        
        pred2 = Prediction(
            class_name="lp",
            box=box2,
            conf=0.90,
            polygon=polygon2
        )
        predictions.append(pred2)
        
        return predictions
    
    def _create_background_predictions(self) -> List[Prediction]:
        """Создает background предсказания (должны быть исключены)"""
        predictions = []
        
        # Background предсказание
        box = Box(
            start=Coords(x=300, y=300),
            end=Coords(x=400, y=400)
        )
        polygon = [
            Coords(x=300, y=300), Coords(x=400, y=300),
            Coords(x=400, y=400), Coords(x=300, y=400)
        ]
        
        pred = Prediction(
            class_name="background",
            box=box,
            conf=0.95,
            polygon=polygon
        )
        predictions.append(pred)
        
        return predictions
    
    def _create_short_segments(self) -> List[Prediction]:
        """Создает короткие сегменты lp (должны быть исключены)"""
        predictions = []
        
        # Короткий сегмент lp (4 точки)
        box = Box(
            start=Coords(x=500, y=500),
            end=Coords(x=600, y=600)
        )
        polygon = [
            Coords(x=500, y=500), Coords(x=600, y=500),
            Coords(x=600, y=600), Coords(x=500, y=600)
        ]
        
        pred = Prediction(
            class_name="lp",
            box=box,
            conf=0.75,
            polygon=polygon
        )
        predictions.append(pred)
        
        # Очень короткий сегмент lp (3 точки)
        box2 = Box(
            start=Coords(x=700, y=700),
            end=Coords(x=800, y=800)
        )
        polygon2 = [
            Coords(x=700, y=700), Coords(x=800, y=700),
            Coords(x=800, y=800)
        ]
        
        pred2 = Prediction(
            class_name="lp",
            box=box2,
            conf=0.70,
            polygon=polygon2
        )
        predictions.append(pred2)
        
        return predictions
    
    def _create_nested_objects(self) -> List[Prediction]:
        """Создает вложенные объекты lp (должны быть исключены)"""
        predictions = []
        
        # Большой lp объект (должен остаться)
        big_box = Box(
            start=Coords(x=900, y=900),
            end=Coords(x=1100, y=1100)
        )
        big_polygon = [
            Coords(x=900, y=900), Coords(x=1100, y=900),
            Coords(x=1100, y=1100), Coords(x=900, y=1100),
            Coords(x=920, y=920), Coords(x=1080, y=920),
            Coords(x=1080, y=1080), Coords(x=920, y=1080)
        ]
        
        big_pred = Prediction(
            class_name="lp",
            box=big_box,
            conf=0.88,
            polygon=big_polygon
        )
        predictions.append(big_pred)
        
        # Малый lp объект внутри большого (вложенный - должен быть исключен)
        small_box = Box(
            start=Coords(x=950, y=950),
            end=Coords(x=1050, y=1050)
        )
        small_polygon = [
            Coords(x=950, y=950), Coords(x=1050, y=950),
            Coords(x=1050, y=1050), Coords(x=950, y=1050),
            Coords(x=970, y=970), Coords(x=1030, y=970),
            Coords(x=1030, y=1030), Coords(x=970, y=1030)
        ]
        
        small_pred = Prediction(
            class_name="lp",
            box=small_box,
            conf=0.82,
            polygon=small_polygon
        )
        predictions.append(small_pred)
        
        return predictions
    
    def _create_other_classes(self) -> List[Prediction]:
        """Создает предсказания других классов"""
        predictions = []
        
        # Mild предсказание
        box = Box(
            start=Coords(x=1200, y=1200),
            end=Coords(x=1300, y=1300)
        )
        polygon = [
            Coords(x=1200, y=1200), Coords(x=1300, y=1200),
            Coords(x=1300, y=1300), Coords(x=1200, y=1300)
        ]
        
        pred = Prediction(
            class_name="mild",
            box=box,
            conf=0.80,
            polygon=polygon
        )
        predictions.append(pred)
        
        # Moderate предсказание
        box2 = Box(
            start=Coords(x=1400, y=1400),
            end=Coords(x=1500, y=1500)
        )
        polygon2 = [
            Coords(x=1400, y=1400), Coords(x=1500, y=1400),
            Coords(x=1500, y=1500), Coords(x=1400, y=1500)
        ]
        
        pred2 = Prediction(
            class_name="moderate",
            box=box2,
            conf=0.85,
            polygon=polygon2
        )
        predictions.append(pred2)
        
        return predictions
    
    def test_background_filtering(self):
        """Тестирует фильтрацию background класса"""
        print("🧪 Тестирование фильтрации background класса...")
        
        predictions = self._create_background_predictions()
        original_count = len(predictions)
        
        filtered = self.merger._filter_background_class(predictions)
        filtered_count = len(filtered)
        
        print(f"   Исходных предсказаний: {original_count}")
        print(f"   После фильтрации: {filtered_count}")
        
        assert filtered_count == 0, f"Background предсказания не были исключены: {filtered_count}"
        print("   ✅ Background класс успешно исключен")
    
    def test_short_segments_filtering(self):
        """Тестирует фильтрацию коротких сегментов"""
        print("🧪 Тестирование фильтрации коротких сегментов...")
        
        predictions = self._create_short_segments()
        original_count = len(predictions)
        
        filtered = self.merger._filter_short_segments(predictions)
        filtered_count = len(filtered)
        
        print(f"   Исходных предсказаний: {original_count}")
        print(f"   После фильтрации: {filtered_count}")
        
        assert filtered_count == 0, f"Короткие сегменты не были исключены: {filtered_count}"
        print("   ✅ Короткие сегменты успешно исключены")
    
    def test_nested_objects_filtering(self):
        """Тестирует фильтрацию вложенных объектов"""
        print("🧪 Тестирование фильтрации вложенных объектов...")
        
        predictions = self._create_nested_objects()
        original_count = len(predictions)
        
        filtered = self.merger._filter_nested_objects(predictions)
        filtered_count = len(filtered)
        
        print(f"   Исходных предсказаний: {original_count}")
        print(f"   После фильтрации: {filtered_count}")
        
        # Должен остаться только один объект (большой)
        assert filtered_count == 1, f"Вложенные объекты не были исключены: {filtered_count}"
        print("   ✅ Вложенные объекты успешно исключены")
    
    def test_improved_iou_filtering(self):
        """Тестирует улучшенную IoU фильтрацию (threshold=0.7)"""
        print("🧪 Тестирование улучшенной IoU фильтрации...")
        
        # Создаем предсказания с разным IoU
        predictions = []
        
        # Предсказание 1
        box1 = Box(start=Coords(x=100, y=100), end=Coords(x=200, y=200))
        pred1 = Prediction(class_name="lp", box=box1, conf=0.9, polygon=None)
        predictions.append(pred1)
        
        # Предсказание 2 с высоким IoU (должно быть исключено)
        box2 = Box(start=Coords(x=150, y=150), end=Coords(x=250, y=250))
        pred2 = Prediction(class_name="lp", box=box2, conf=0.8, polygon=None)
        predictions.append(pred2)
        
        # Предсказание 3 с низким IoU (должно остаться)
        box3 = Box(start=Coords(x=300, y=300), end=Coords(x=400, y=400))
        pred3 = Prediction(class_name="lp", box=box3, conf=0.7, polygon=None)
        predictions.append(pred3)
        
        original_count = len(predictions)
        filtered = self.merger.filter_by_improved_iou(predictions)
        filtered_count = len(filtered)
        
        print(f"   Исходных предсказаний: {original_count}")
        print(f"   После фильтрации: {filtered_count}")
        
        # Должно остаться 2 предсказания (1 и 3)
        assert filtered_count == 2, f"Неправильная IoU фильтрация: {filtered_count}"
        print("   ✅ IoU фильтрация работает корректно")
    
    def test_complete_merger_pipeline(self):
        """Тестирует полный pipeline улучшенного объединения"""
        print("🧪 Тестирование полного pipeline улучшенного объединения...")
        
        predictions = self.create_test_predictions()
        original_count = len(predictions)
        
        print(f"   Исходных предсказаний: {original_count}")
        
        # Запускаем полный pipeline
        merged_predictions = self.merger.merge_predictions(predictions)
        merged_count = len(merged_predictions)
        
        print(f"   После объединения: {merged_count}")
        
        # Получаем статистику
        stats = self.merger.get_filtering_statistics(predictions, merged_predictions)
        
        print(f"   Исключено предсказаний: {stats['filtered_out']}")
        print(f"   Коэффициент фильтрации: {stats['filtering_ratio']:.2%}")
        
        # Проверяем, что background класс исключен
        background_count = sum(1 for p in merged_predictions if p.class_name == "background")
        assert background_count == 0, f"Background класс не был исключен: {background_count}"
        
        # Проверяем, что короткие сегменты lp исключены
        short_lp_count = sum(1 for p in merged_predictions 
                           if p.class_name == "lp" and p.polygon and len(p.polygon) < 8)
        assert short_lp_count == 0, f"Короткие сегменты lp не были исключены: {short_lp_count}"
        
        print("   ✅ Полный pipeline работает корректно")
        
        return stats
    
    def run_all_tests(self):
        """Запускает все тесты"""
        print("🚀 Запуск тестов улучшенного алгоритма объединения")
        print("=" * 60)
        
        try:
            # Тест 1: Фильтрация background класса
            self.test_background_filtering()
            print()
            
            # Тест 2: Фильтрация коротких сегментов
            self.test_short_segments_filtering()
            print()
            
            # Тест 3: Фильтрация вложенных объектов
            self.test_nested_objects_filtering()
            print()
            
            # Тест 4: Улучшенная IoU фильтрация
            self.test_improved_iou_filtering()
            print()
            
            # Тест 5: Полный pipeline
            stats = self.test_complete_merger_pipeline()
            print()
            
            print("📊 Итоговая статистика:")
            print("-" * 40)
            print(f"Исходных предсказаний: {stats['total_original']}")
            print(f"После фильтрации: {stats['total_filtered']}")
            print(f"Исключено: {stats['filtered_out']}")
            print(f"Коэффициент фильтрации: {stats['filtering_ratio']:.2%}")
            
            print("\n✅ Все тесты пройдены успешно!")
            
        except Exception as e:
            print(f"\n❌ Ошибка тестирования: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Основная функция для запуска тестов"""
    tester = ImprovedMergerTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Детектор биоптатов для WSI YOLO Pipeline
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BiopsyRegion:
    """Область биоптата"""
    id: int
    name: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int

@dataclass
class GridConfig:
    """Конфигурация сетки"""
    step_x: int
    step_y: int
    cell_width: int
    cell_height: int

class BiopsyDetector:
    """Детектор биоптатов для WSI"""
    
    def __init__(self, config_path: str = "manual_biopsy_analysis.json"):
        """
        Инициализация детектора биоптатов
        
        Args:
            config_path: Путь к файлу конфигурации с результатами анализа
        """
        self.config_path = config_path
        self.biopsy_regions: List[BiopsyRegion] = []
        self.grid_config: Optional[GridConfig] = None
        self.wsi_size: Tuple[int, int] = (136192, 77312)
        
        self._load_config()
    
    def _load_config(self):
        """Загружает конфигурацию из файла"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                manual_analysis = config.get('manual_analysis', {})
                
                # Загружаем биоптаты
                biopsy_data = manual_analysis.get('biopsy_regions', [])
                self.biopsy_regions = [
                    BiopsyRegion(
                        id=region['id'],
                        name=region['name'],
                        x_min=region['x_min'],
                        y_min=region['y_min'],
                        x_max=region['x_max'],
                        y_max=region['y_max'],
                        width=region['width'],
                        height=region['height']
                    )
                    for region in biopsy_data
                ]
                
                # Загружаем конфигурацию сетки
                grid_data = manual_analysis.get('recommended_grid', {})
                if grid_data:
                    self.grid_config = GridConfig(
                        step_x=grid_data['step_x'],
                        step_y=grid_data['step_y'],
                        cell_width=grid_data['cell_width'],
                        cell_height=grid_data['cell_height']
                    )
                
                print(f"✅ Загружено {len(self.biopsy_regions)} биоптатов")
                print(f"✅ Сетка: {self.grid_config.step_x}x{self.grid_config.step_y}")
                
            else:
                print(f"⚠️ Файл конфигурации не найден: {self.config_path}")
                print("💡 Запустите manual_biopsy_analysis.py для создания конфигурации")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
    
    def get_biopsy_count(self) -> int:
        """Возвращает количество биоптатов"""
        return len(self.biopsy_regions)
    
    def get_biopsy_regions(self) -> List[BiopsyRegion]:
        """Возвращает список биоптатов"""
        return self.biopsy_regions
    
    def get_grid_config(self) -> Optional[GridConfig]:
        """Возвращает конфигурацию сетки"""
        return self.grid_config
    
    def get_biopsy_by_id(self, biopsy_id: int) -> Optional[BiopsyRegion]:
        """Возвращает биоптат по ID"""
        for region in self.biopsy_regions:
            if region.id == biopsy_id:
                return region
        return None
    
    def get_biopsy_at_position(self, x: int, y: int) -> Optional[BiopsyRegion]:
        """Возвращает биоптат, содержащий указанную позицию"""
        for region in self.biopsy_regions:
            if (region.x_min <= x <= region.x_max and 
                region.y_min <= y <= region.y_max):
                return region
        return None
    
    def get_grid_cell_for_position(self, x: int, y: int) -> Tuple[int, int]:
        """Возвращает координаты ячейки сетки для указанной позиции"""
        if not self.grid_config:
            return (0, 0)
        
        cell_x = x // self.grid_config.step_x
        cell_y = y // self.grid_config.step_y
        return (cell_x, cell_y)
    
    def get_biopsy_for_detailed_analysis(self, biopsy_id: int = 1) -> Optional[BiopsyRegion]:
        """
        Возвращает биоптат для детального анализа
        
        Args:
            biopsy_id: ID биоптата для анализа (по умолчанию 1)
        
        Returns:
            BiopsyRegion или None
        """
        return self.get_biopsy_by_id(biopsy_id)
    
    def get_speedup_factor(self) -> int:
        """Возвращает коэффициент ускорения (количество биоптатов)"""
        return len(self.biopsy_regions)
    
    def is_position_in_biopsy(self, x: int, y: int) -> bool:
        """Проверяет, находится ли позиция в каком-либо биоптате"""
        return self.get_biopsy_at_position(x, y) is not None
    
    def get_biopsy_statistics(self) -> Dict:
        """Возвращает статистику по биоптатам"""
        if not self.biopsy_regions:
            return {}
        
        # Вычисляем статистику
        total_area = sum(region.width * region.height for region in self.biopsy_regions)
        avg_width = sum(region.width for region in self.biopsy_regions) / len(self.biopsy_regions)
        avg_height = sum(region.height for region in self.biopsy_regions) / len(self.biopsy_regions)
        
        return {
            "biopsy_count": len(self.biopsy_regions),
            "total_area": total_area,
            "average_width": avg_width,
            "average_height": avg_height,
            "speedup_factor": self.get_speedup_factor(),
            "grid_step": f"{self.grid_config.step_x}x{self.grid_config.step_y}" if self.grid_config else "N/A"
        }
    
    def create_biopsy_mask(self, wsi_size: Tuple[int, int]) -> List[List[bool]]:
        """
        Создает маску биоптатов для WSI
        
        Args:
            wsi_size: Размеры WSI (width, height)
        
        Returns:
            Маска биоптатов (True если позиция в биоптате)
        """
        width, height = wsi_size
        mask = [[False for _ in range(width)] for _ in range(height)]
        
        for region in self.biopsy_regions:
            for x in range(region.x_min, min(region.x_max, width)):
                for y in range(region.y_min, min(region.y_max, height)):
                    mask[y][x] = True
        
        return mask
    
    def get_optimization_recommendations(self) -> Dict:
        """Возвращает рекомендации по оптимизации pipeline"""
        stats = self.get_biopsy_statistics()
        
        return {
            "detailed_analysis": {
                "recommended_biopsy_id": 1,
                "biopsy_name": self.get_biopsy_by_id(1).name if self.get_biopsy_by_id(1) else "N/A",
                "reason": "Анализ одного биоптата для экстраполяции на остальные"
            },
            "speed_optimization": {
                "speedup_factor": stats["speedup_factor"],
                "time_reduction": f"{100 - (100 / stats['speedup_factor']):.1f}%",
                "reason": f"Обработка 1 из {stats['biopsy_count']} биоптатов"
            },
            "grid_optimization": {
                "grid_step": self.grid_config.step_x if self.grid_config else "N/A",
                "cell_size": f"{self.grid_config.cell_width}x{self.grid_config.cell_height}" if self.grid_config else "N/A",
                "reason": "Сетка без пересечений с биоптатами"
            }
        }

def main():
    """Тестирование детектора биоптатов"""
    print("🧪 Тестирование детектора биоптатов")
    print("=" * 50)
    
    detector = BiopsyDetector()
    
    if detector.biopsy_regions:
        print(f"✅ Загружено {detector.get_biopsy_count()} биоптатов")
        
        # Статистика
        stats = detector.get_biopsy_statistics()
        print(f"\n📊 Статистика:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Рекомендации
        recommendations = detector.get_optimization_recommendations()
        print(f"\n🎯 Рекомендации по оптимизации:")
        for category, recs in recommendations.items():
            print(f"   {category}:")
            for key, value in recs.items():
                print(f"     {key}: {value}")
        
        # Тестирование позиций
        print(f"\n🔍 Тестирование позиций:")
        test_positions = [(50000, 50000), (10000, 10000), (60000, 30000)]
        for x, y in test_positions:
            biopsy = detector.get_biopsy_at_position(x, y)
            if biopsy:
                print(f"   Позиция ({x}, {y}): {biopsy.name}")
            else:
                print(f"   Позиция ({x}, {y}): вне биоптатов")
    
    else:
        print("❌ Биоптаты не загружены")
        print("💡 Запустите manual_biopsy_analysis.py для создания конфигурации")

if __name__ == "__main__":
    main()

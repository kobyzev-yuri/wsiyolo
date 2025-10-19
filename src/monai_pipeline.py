"""
MONAI pipeline для работы с WSI и извлечения патчей.
Использует GridPatchd с threshold для отсева пустых патчей.
"""

import numpy as np
from typing import List, Tuple, Optional
from monai.data import CuCIMWSIReader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, GridPatchd
import torch

from .data_structures import PatchInfo, WSIInfo, Coords


class WSIPipeline:
    """Pipeline для обработки WSI с использованием MONAI"""
    
    def __init__(self, tile_size: int = 512, overlap_ratio: float = 0.5):
        """
        Инициализация pipeline
        
        Args:
            tile_size: Размер патча
            overlap_ratio: Коэффициент перекрытия (0.5 = 50% перекрытие)
        """
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.step_size = int(tile_size * (1 - overlap_ratio))
        
        # Создаем transforms для MONAI
        self.transforms = Compose([
            LoadImaged(keys=["image"], reader=CuCIMWSIReader, level=0),
            EnsureChannelFirstd(keys=["image"]),
            GridPatchd(
                keys=["image"],
                patch_size=(tile_size, tile_size),
                threshold=0.999 * 3 * 255 * tile_size * tile_size,
                pad_mode=None,
                constant_values=255,
            )
        ])
    
    def load_wsi(self, wsi_path: str) -> WSIInfo:
        """
        Загружает информацию о WSI
        
        Args:
            wsi_path: Путь к WSI файлу
            
        Returns:
            WSIInfo: Информация о WSI
        """
        try:
            # Создаем reader для получения метаданных
            reader = CuCIMWSIReader(level=0)
            
            # Получаем метаданные без загрузки изображения
            wsi = reader.read(wsi_path)
            
            # Извлекаем информацию о размерах
            width = wsi.shape[1] if len(wsi.shape) > 1 else 0
            height = wsi.shape[0] if len(wsi.shape) > 0 else 0
            
            # Получаем количество уровней
            try:
                levels = wsi.resolutions["level_count"] if hasattr(wsi, 'resolutions') else 1
            except:
                levels = 1
            
            # Получаем MPP
            try:
                mpp = wsi.metadata.get('mpp', None) if hasattr(wsi, 'metadata') else None
            except:
                mpp = None
            
            return WSIInfo(
                path=wsi_path,
                width=width,
                height=height,
                levels=levels,
                level_downsamples=[1.0],  # По умолчанию
                mpp=mpp
            )
            
        except Exception as e:
            print(f"⚠️  Ошибка загрузки WSI: {e}")
            # Возвращаем базовую информацию
            return WSIInfo(
                path=wsi_path,
                width=0,
                height=0,
                levels=1,
                level_downsamples=[1.0],
                mpp=None
            )
    
    def extract_patches(self, wsi_path: str) -> List[PatchInfo]:
        """
        Извлекает патчи из WSI с использованием GridPatchd
        
        Args:
            wsi_path: Путь к WSI файлу
            
        Returns:
            List[PatchInfo]: Список патчей с информацией
        """
        # Подготавливаем данные для transforms
        data = {"image": wsi_path}
        
        # Применяем transforms
        result = self.transforms(data)
        
        patches = []
        patch_id = 0
        
        # Извлекаем патчи из результата
        if "image" in result:
            patches_array = result["image"]
            
            # Получаем координаты патчей
            if "image_location" in result:
                locations = result["image_location"]
                
                for i, (patch_img, location) in enumerate(zip(patches_array, locations)):
                    # location содержит координаты патча
                    x, y = location[0], location[1]  # Координаты в WSI
                    
                    # Создаем PatchInfo
                    patch_info = PatchInfo(
                        patch_id=patch_id,
                        x=x,
                        y=y,
                        size=self.tile_size,
                        image=patch_img,
                        has_tissue=True  # GridPatchd уже отфильтровал пустые патчи
                    )
                    
                    patches.append(patch_info)
                    patch_id += 1
        
        return patches
    
    def extract_patches_with_overlap(self, wsi_path: str) -> List[PatchInfo]:
        """
        Извлекает патчи с перекрытием вручную (альтернативный метод)
        
        Args:
            wsi_path: Путь к WSI файлу
            
        Returns:
            List[PatchInfo]: Список патчей с информацией
        """
        # Загружаем WSI
        reader = CuCIMWSIReader(level=0)
        img_array, metadata = reader.get_data(wsi_path)
        
        patches = []
        patch_id = 0
        
        # Вычисляем сетку патчей с перекрытием
        height, width = img_array.shape[:2]
        
        for y in range(0, height - self.tile_size + 1, self.step_size):
            for x in range(0, width - self.tile_size + 1, self.step_size):
                # Извлекаем патч
                patch_img = img_array[y:y+self.tile_size, x:x+self.tile_size]
                
                # Проверяем, содержит ли патч ткань (простая проверка)
                has_tissue = self._has_tissue(patch_img)
                
                if has_tissue:
                    patch_info = PatchInfo(
                        patch_id=patch_id,
                        x=x,
                        y=y,
                        size=self.tile_size,
                        image=patch_img,
                        has_tissue=True
                    )
                    patches.append(patch_info)
                    patch_id += 1
        
        return patches
    
    def _has_tissue(self, patch: np.ndarray) -> bool:
        """
        Простая проверка наличия ткани в патче
        
        Args:
            patch: Изображение патча
            
        Returns:
            bool: True если патч содержит ткань
        """
        # Простая проверка: если средняя яркость меньше 240, то есть ткань
        if len(patch.shape) == 3:
            mean_brightness = np.mean(patch)
        else:
            mean_brightness = np.mean(patch)
        
        return mean_brightness < 240  # Порог для определения ткани
    
    def get_patch_coordinates(self, wsi_info: WSIInfo) -> List[Tuple[int, int]]:
        """
        Вычисляет координаты всех патчей в WSI
        
        Args:
            wsi_info: Информация о WSI
            
        Returns:
            List[Tuple[int, int]]: Список координат (x, y) патчей
        """
        coordinates = []
        
        for y in range(0, wsi_info.height - self.tile_size + 1, self.step_size):
            for x in range(0, wsi_info.width - self.tile_size + 1, self.step_size):
                coordinates.append((x, y))
        
        return coordinates

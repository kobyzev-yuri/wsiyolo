"""
WSI Patch Loader с использованием MONAI GridPatchd.
Извлекает патчи из WSI с отсевом пустых областей.
"""

import numpy as np
from typing import List, Tuple, Optional
from monai.data import CuCIMWSIReader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, GridPatchd
import torch

from .data_structures import PatchInfo, WSIInfo, Coords


class WSIPatchLoader:
    """Загрузчик патчей из WSI с использованием MONAI GridPatchd"""
    
    def __init__(self, tile_size: int = 512, overlap_ratio: float = 0.5):
        """
        Инициализация загрузчика
        
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
    
    def load_wsi_info(self, wsi_path: str) -> WSIInfo:
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
            
            # Получаем базовую информацию о WSI
            wsi = reader.read(wsi_path)
            
            # Извлекаем размеры
            if hasattr(wsi, 'shape'):
                height, width = wsi.shape[:2]
            else:
                # Fallback - загружаем небольшой кусок для определения размера
                small_patch = reader.get_data(wsi_path, size=(100, 100))[0]
                height, width = small_patch.shape[:2]
                # Приблизительно вычисляем полный размер
                height *= 10  # Примерное масштабирование
                width *= 10
            
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
                level_downsamples=[1.0],
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
        print(f"🔍 Извлечение патчей из WSI: {wsi_path}")
        
        try:
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
                
                print(f"✅ Извлечено патчей: {len(patches)}")
            else:
                print("⚠️  Патчи не найдены в результате")
            
            return patches
            
        except Exception as e:
            print(f"❌ Ошибка извлечения патчей: {e}")
            return []
    
    def extract_patches_manual(self, wsi_path: str) -> List[PatchInfo]:
        """
        Альтернативный метод извлечения патчей вручную с перекрытием
        
        Args:
            wsi_path: Путь к WSI файлу
            
        Returns:
            List[PatchInfo]: Список патчей с информацией
        """
        print(f"🔍 Ручное извлечение патчей из WSI: {wsi_path}")
        
        try:
            # Загружаем WSI
            reader = CuCIMWSIReader(level=0)
            img_array, metadata = reader.get_data(wsi_path)
            
            patches = []
            patch_id = 0
            
            # Вычисляем сетку патчей с перекрытием
            height, width = img_array.shape[:2]
            
            print(f"📊 Размер WSI: {width}x{height}")
            print(f"🔧 Параметры: tile_size={self.tile_size}, step_size={self.step_size}")
            
            # Извлекаем патчи с перекрытием
            for y in range(0, height - self.tile_size + 1, self.step_size):
                for x in range(0, width - self.tile_size + 1, self.step_size):
                    # Извлекаем патч
                    patch_img = img_array[y:y+self.tile_size, x:x+self.tile_size]
                    
                    # Проверяем, содержит ли патч ткань
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
            
            print(f"✅ Извлечено патчей с тканью: {len(patches)}")
            return patches
            
        except Exception as e:
            print(f"❌ Ошибка ручного извлечения патчей: {e}")
            return []
    
    def _has_tissue(self, patch: np.ndarray) -> bool:
        """
        Проверяет наличие ткани в патче
        
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
    
    def save_patch(self, patch_info: PatchInfo, output_path: str):
        """
        Сохраняет патч в файл
        
        Args:
            patch_info: Информация о патче
            output_path: Путь для сохранения
        """
        try:
            import cv2
            
            # Конвертируем в формат для сохранения
            if len(patch_info.image.shape) == 3:
                # RGB изображение
                img_to_save = cv2.cvtColor(patch_info.image, cv2.COLOR_RGB2BGR)
            else:
                # Grayscale
                img_to_save = patch_info.image
            
            cv2.imwrite(output_path, img_to_save)
            
        except Exception as e:
            print(f"⚠️  Ошибка сохранения патча: {e}")

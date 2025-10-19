"""
Простой и надежный patch loader для WSI.
Использует cuCIM напрямую для извлечения патчей.
"""

import numpy as np
from typing import List, Tuple
import cucim
from cucim import CuImage

from data_structures import PatchInfo, WSIInfo, Coords


class SimplePatchLoader:
    """Простой загрузчик патчей из WSI"""
    
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
    
    def load_wsi_info(self, wsi_path: str) -> WSIInfo:
        """
        Загружает информацию о WSI
        
        Args:
            wsi_path: Путь к WSI файлу
            
        Returns:
            WSIInfo: Информация о WSI
        """
        try:
            # Загружаем WSI с помощью cuCIM
            wsi = CuImage(wsi_path)
            
            # Получаем размеры
            width = wsi.shape[1]
            height = wsi.shape[0]
            
            # Получаем количество уровней
            levels = wsi.resolutions["level_count"]
            
            # Получаем MPP
            mpp = wsi.metadata.get('mpp', None)
            
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
    
    def extract_patches(self, wsi_path: str, max_patches: int = None) -> List[PatchInfo]:
        """
        Извлекает патчи из WSI
        
        Args:
            wsi_path: Путь к WSI файлу
            max_patches: Максимальное количество патчей
            
        Returns:
            List[PatchInfo]: Список патчей с информацией
        """
        print(f"🔍 Извлечение патчей из WSI: {wsi_path}")
        
        try:
            # Загружаем WSI
            wsi = CuImage(wsi_path)
            
            patches = []
            patch_id = 0
            
            # Вычисляем сетку патчей с перекрытием
            height, width = wsi.shape[:2]
            
            print(f"📊 Размер WSI: {width}x{height}")
            print(f"🔧 Параметры: tile_size={self.tile_size}, step_size={self.step_size}")
            
            # Вычисляем общее количество патчей для прогресс-бара
            total_patches = 0
            for y in range(0, height - self.tile_size + 1, self.step_size):
                for x in range(0, width - self.tile_size + 1, self.step_size):
                    total_patches += 1
            
            print(f"   Всего патчей для проверки: {total_patches}")
            
            # Извлекаем патчи с перекрытием
            from tqdm import tqdm
            
            patch_coords = []
            for y in range(0, height - self.tile_size + 1, self.step_size):
                for x in range(0, width - self.tile_size + 1, self.step_size):
                    patch_coords.append((x, y))
            
            for x, y in tqdm(patch_coords, desc="Извлечение патчей"):
                # Извлекаем патч
                patch_img = wsi.read_region((x, y), (self.tile_size, self.tile_size))
                
                # Конвертируем в numpy array и RGB формат
                if hasattr(patch_img, 'numpy'):
                    patch_img = patch_img.numpy()
                else:
                    patch_img = np.array(patch_img)
                
                # Убеждаемся, что это RGB изображение
                if len(patch_img.shape) == 3 and patch_img.shape[2] == 4:
                    # RGBA -> RGB
                    patch_img = patch_img[:, :, :3]
                elif len(patch_img.shape) == 2:
                    # Grayscale -> RGB
                    patch_img = np.stack([patch_img] * 3, axis=-1)
                
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
                    
                    # Ограничиваем количество патчей
                    if max_patches and len(patches) >= max_patches:
                        break
            
            print(f"✅ Извлечено патчей с тканью: {len(patches)}")
            return patches
            
        except Exception as e:
            print(f"❌ Ошибка извлечения патчей: {e}")
            return []
    
    def _has_tissue(self, patch: np.ndarray) -> bool:
        """
        Проверяет наличие ткани в патче
        
        Args:
            patch: Изображение патча
            
        Returns:
            bool: True если патч содержит ткань
        """
        # Более строгая проверка ткани
        if len(patch.shape) == 3:
            # RGB изображение - проверяем каждый канал
            mean_brightness = np.mean(patch)
            # Дополнительная проверка: если все каналы очень светлые, то это фон
            if np.all(patch > 240):
                return False
        else:
            mean_brightness = np.mean(patch)
        
        # Проверяем стандартное отклонение (ткань должна иметь вариацию)
        std_dev = np.std(patch)
        
        # Патч содержит ткань если:
        # 1. Средняя яркость не слишком высокая
        # 2. Есть достаточная вариация пикселей
        has_tissue = (mean_brightness < 240) and (std_dev > 10)
        
        return has_tissue
    
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

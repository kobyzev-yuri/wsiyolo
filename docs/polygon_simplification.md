# 🎯 Алгоритм упрощения полигонов

## Обзор

Алгоритм упрощения полигонов в WSI YOLO pipeline предназначен для уменьшения количества точек в сегментационных масках до максимум 60 точек, сохраняя при этом важные геометрические особенности объектов.

## 🔧 Параметры алгоритма

### Основные параметры

```python
# Параметры упрощения
max_points = 60                    # Максимальное количество точек
min_tolerance = 0.1               # Минимальный tolerance для Douglas-Peucker
max_tolerance = 10.0              # Максимальный tolerance
max_iterations = 10               # Максимум итераций бинарного поиска
target_ratio = 0.8                # Целевое соотношение (80% от max_points)
```

### Параметры контуров

```python
# Параметры извлечения контуров
contour_threshold = 0.5           # Порог для find_contours
preserve_topology = True          # Сохранение топологии при simplify
```

## 📋 Этапы алгоритма

### 1. Извлечение контуров

```python
def _mask_to_polygon(self, mask: np.ndarray, offset_x: int, offset_y: int) -> List[Coords]:
    # Находим контуры из маски
    contours = measure.find_contours(mask, 0.5)
    
    # Берем самый большой контур
    largest_contour = max(contours, key=len)
    
    # Преобразуем в абсолютные координаты
    coords = []
    for point in largest_contour:
        x, y = point[1], point[0]  # skimage использует (row, col)
        coords.append((offset_x + x, offset_y + y))
```

### 2. Создание Shapely полигона

```python
# Создаем Shapely полигон для умного упрощения
if len(coords) >= 3:
    poly = Polygon(coords)
    if poly.is_valid:
        # Умное упрощение до максимум 60 точек
        simplified = self._smart_simplify_polygon(poly, max_points=60)
```

### 3. Умное упрощение

```python
def _smart_simplify_polygon(self, polygon: Polygon, max_points: int = 60) -> Polygon:
    current_points = len(current_poly.exterior.coords)
    
    if current_points <= max_points:
        return current_poly
    
    # Адаптивное упрощение с бинарным поиском tolerance
    min_tolerance = 0.1
    max_tolerance = 10.0
    best_poly = current_poly
```

### 4. Бинарный поиск оптимального tolerance

```python
# Бинарный поиск оптимального tolerance
for _ in range(10):  # Максимум 10 итераций
    tolerance = (min_tolerance + max_tolerance) / 2
    simplified = current_poly.simplify(tolerance, preserve_topology=True)
    
    if simplified.is_valid and len(simplified.exterior.coords) > 3:
        points_count = len(simplified.exterior.coords)
        
        if points_count <= max_points:
            # Нашли подходящий результат
            best_poly = simplified
            min_tolerance = tolerance
            if points_count >= max_points * 0.8:  # Если достаточно близко к цели
                break
        else:
            # Нужно больше упрощения
            max_tolerance = tolerance
    else:
        # Упрощение слишком агрессивное
        max_tolerance = tolerance
```

### 5. Fallback: равномерная выборка

```python
# Если все еще слишком много точек, используем равномерную выборку
if len(best_poly.exterior.coords) > max_points:
    coords = list(best_poly.exterior.coords)
    step = len(coords) // max_points
    sampled_coords = coords[::max(1, step)]
    
    # Создаем новый полигон из выбранных точек
    if len(sampled_coords) >= 3:
        sampled_poly = Polygon(sampled_coords)
        if sampled_poly.is_valid:
            best_poly = sampled_poly
```

## ⚠️ Известные проблемы

### 1. Потеря топологии

**Проблема**: Равномерная выборка может нарушить структуру полигона
```python
# Проблемный код
sampled_coords = coords[::max(1, step)]
```

**Последствия**:
- Полигон может стать самопересекающимся
- Потеря важных геометрических особенностей
- Некорректные результаты при объединении

### 2. Искажение формы

**Проблема**: Агрессивное упрощение может изменить форму объекта
```python
# Слишком высокий tolerance
tolerance = 10.0  # Может быть слишком агрессивным
```

**Последствия**:
- Потеря мелких деталей
- Изменение пропорций объекта
- Неточные границы сегментации

### 3. Невалидные полигоны

**Проблема**: После упрощения полигон может стать некорректным
```python
# Проверка валидности
if simplified.is_valid and len(simplified.exterior.coords) > 3:
    # Может быть недостаточно
```

**Последствия**:
- Ошибки при геометрических операциях
- Сбой объединения полигонов
- Некорректные результаты

### 4. Неоптимальный tolerance

**Проблема**: Бинарный поиск может не найти лучшее значение
```python
# Ограниченное количество итераций
for _ in range(10):  # Может быть недостаточно
```

**Последствия**:
- Субоптимальное упрощение
- Лишние точки или потеря деталей
- Неэффективное использование памяти

## 🔧 Рекомендации по улучшению

### 1. Адаптивный tolerance

```python
def calculate_adaptive_tolerance(polygon: Polygon, max_points: int) -> float:
    """Вычисляет адаптивный tolerance на основе площади полигона"""
    area = polygon.area
    perimeter = polygon.length
    
    # Базовый tolerance на основе размера полигона
    base_tolerance = min(area / 1000, perimeter / 100)
    
    # Корректировка на основе сложности формы
    complexity = len(polygon.exterior.coords) / max_points
    adaptive_tolerance = base_tolerance * complexity
    
    return max(0.1, min(10.0, adaptive_tolerance))
```

### 2. Сохранение ключевых точек

```python
def preserve_key_points(coords: List[Tuple], max_points: int) -> List[Tuple]:
    """Сохраняет углы и точки с высокой кривизной"""
    if len(coords) <= max_points:
        return coords
    
    # Вычисляем кривизну для каждой точки
    curvatures = calculate_curvature(coords)
    
    # Сортируем по кривизне и сохраняем топ-N
    key_indices = sorted(range(len(curvatures)), 
                        key=lambda i: curvatures[i], 
                        reverse=True)[:max_points]
    
    return [coords[i] for i in sorted(key_indices)]
```

### 3. Метрики качества

```python
def calculate_simplification_quality(original: Polygon, simplified: Polygon) -> dict:
    """Вычисляет метрики качества упрощения"""
    area_ratio = simplified.area / original.area
    perimeter_ratio = simplified.length / original.length
    
    # Проверка на значительные изменения
    quality_score = 1.0 - abs(1.0 - area_ratio) - abs(1.0 - perimeter_ratio)
    
    return {
        'area_ratio': area_ratio,
        'perimeter_ratio': perimeter_ratio,
        'quality_score': quality_score,
        'is_acceptable': quality_score > 0.8
    }
```

### 4. Альтернативные стратегии

```python
def alternative_simplification_strategies(polygon: Polygon, max_points: int) -> List[Polygon]:
    """Пробует разные стратегии упрощения"""
    strategies = []
    
    # 1. Douglas-Peucker с разными tolerance
    for tolerance in [0.1, 0.5, 1.0, 2.0, 5.0]:
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        if len(simplified.exterior.coords) <= max_points:
            strategies.append(simplified)
    
    # 2. Равномерная выборка с разными шагами
    coords = list(polygon.exterior.coords)
    for step in range(1, len(coords) // max_points + 1):
        sampled = coords[::step]
        if len(sampled) <= max_points:
            try:
                sampled_poly = Polygon(sampled)
                if sampled_poly.is_valid:
                    strategies.append(sampled_poly)
            except:
                continue
    
    return strategies
```

## 🧪 Тестирование и отладка

### Логирование для отладки

```python
def debug_simplification(polygon: Polygon, max_points: int) -> dict:
    """Детальная отладка процесса упрощения"""
    debug_info = {
        'original_points': len(polygon.exterior.coords),
        'target_points': max_points,
        'original_area': polygon.area,
        'original_perimeter': polygon.length,
        'steps': []
    }
    
    # Логируем каждый шаг
    for tolerance in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        step_info = {
            'tolerance': tolerance,
            'points': len(simplified.exterior.coords),
            'area_ratio': simplified.area / polygon.area,
            'is_valid': simplified.is_valid
        }
        debug_info['steps'].append(step_info)
    
    return debug_info
```

### Валидация результатов

```python
def validate_simplified_polygon(original: Polygon, simplified: Polygon) -> bool:
    """Проверяет качество упрощенного полигона"""
    
    # 1. Проверка валидности
    if not simplified.is_valid:
        return False
    
    # 2. Проверка минимального количества точек
    if len(simplified.exterior.coords) < 3:
        return False
    
    # 3. Проверка сохранения основных характеристик
    area_ratio = simplified.area / original.area
    if area_ratio < 0.5 or area_ratio > 1.5:  # Допустимое изменение площади ±50%
        return False
    
    # 4. Проверка на самопересечения
    if not simplified.is_simple:
        return False
    
    return True
```

## 📊 Мониторинг производительности

### Метрики для отслеживания

```python
class SimplificationMetrics:
    def __init__(self):
        self.total_polygons = 0
        self.successful_simplifications = 0
        self.failed_simplifications = 0
        self.average_reduction_ratio = 0.0
        self.processing_times = []
    
    def record_simplification(self, original_points: int, final_points: int, 
                           processing_time: float, success: bool):
        """Записывает метрики упрощения"""
        self.total_polygons += 1
        
        if success:
            self.successful_simplifications += 1
            reduction_ratio = (original_points - final_points) / original_points
            self.average_reduction_ratio = (
                (self.average_reduction_ratio * (self.successful_simplifications - 1) + 
                 reduction_ratio) / self.successful_simplifications
            )
        else:
            self.failed_simplifications += 1
        
        self.processing_times.append(processing_time)
    
    def get_summary(self) -> dict:
        """Возвращает сводку метрик"""
        return {
            'total_polygons': self.total_polygons,
            'success_rate': self.successful_simplifications / self.total_polygons,
            'average_reduction_ratio': self.average_reduction_ratio,
            'average_processing_time': sum(self.processing_times) / len(self.processing_times),
            'failed_simplifications': self.failed_simplifications
        }
```

## 🚀 Практические рекомендации

### 1. Настройка параметров

- **Для простых форм**: `max_points = 30-40`
- **Для сложных форм**: `max_points = 60-80`
- **Для критически важных деталей**: `max_points = 100+`

### 2. Мониторинг качества

- Регулярно проверяйте метрики качества
- Визуализируйте результаты упрощения
- Сравнивайте с исходными полигонами

### 3. Отладка проблем

- Используйте детальное логирование
- Анализируйте случаи неудачного упрощения
- Тестируйте на различных типах полигонов

### 4. Оптимизация производительности

- Кэшируйте результаты для похожих полигонов
- Используйте параллельную обработку
- Мониторьте использование памяти

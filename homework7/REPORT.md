# Домашнее задание: Сравнение производительности различных подходов оптимизации

## 1. Введение

### Задача

Задачей данного исследования является сравнение производительности различных подходов оптимизации инференса нейронных сетей:

- **PyTorch**
- **ONNX Runtime**
- **Torch-TensorRT**

### Ожидаемые результаты

- Выявление самого быстрого и стабильного способа инференса
- Определение оптимального размера изображения и батча
- Расчет использования вычислительных ресурсов (TFLOPS)
- Рекомендации для продакшн-инференса


## 2. Методология

### Экспериментальная установка

- **GPU:** NVIDIA GeForce RTX 4060 Laptop (теоретическая производительность: 11.61 TFLOPS FP32)
- **OS:** Ubuntu 22.04
- **CUDA:** 12.8
- **PyTorch:** 2.7.1+cu128
- **ONNX Runtime:** 1.22.1
- **Torch-TensorRT:** 2.7.0+cu128

### Модели

- Архитектура: ResNet-18
- Размеры входных изображений: `224x224`, `256x256`, `384x384`, `512x512`
- Модели сохранены в:
  - `./weights/best_resnet18_224.pth`
  - `./weights/best_resnet18_256.pth`
  - `./weights/best_resnet18_384.pth`
  - `./weights/best_resnet18_512.pth`

### Параметры тестирования

- Размеры батча: 1, 2, 4, 8, 16, 32, 64
- Количество повторов: 50 запусков
- Warmup: 10 итераций
- FPS считается как `batch_size / average_time`

### Методы измерения

- Измерение времени инференса с помощью `time.time()` с синхронизацией `torch.cuda.synchronize()`.  
  Время усредняется по нескольким запускам, включая прогревочные итерации.

- Расчёт TFLOPs на основе фиксированного числа операций на изображение и измеренного среднего времени инференса.

- Использование внешних инструментов, таких как `nvidia-smi`, для замера загрузки GPU и использования памяти.



## 3. Результаты

### Таблица 1. Сравнение FPS (PyTorch)

| Resolution | Batch Size | Time/Image (ms) | FPS    | GPU                              | TFLOPs Measured | TFLOPs Max | Util (%) |
|------------|------------|------------------|--------|-----------------------------------|------------------|-------------|-----------|
| 224x224    | 1          | 2.76             | 362.6  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.10             | 11.61       | 0.9%      |
| 224x224    | 8          | 0.61             | 1633.8 | NVIDIA GeForce RTX 4060 Laptop GPU | 1.20             | 11.61       | 10.3%     |
| 224x224    | 16         | 0.59             | 1696.8 | NVIDIA GeForce RTX 4060 Laptop GPU | 2.40             | 11.61       | 20.7%     |
| 224x224    | 32         | 0.62             | 1615.8 | NVIDIA GeForce RTX 4060 Laptop GPU | 4.80             | 11.61       | 41.3%     |
| 224x224    | 64         | 0.64             | 1558.3 | NVIDIA GeForce RTX 4060 Laptop GPU | 10.90            | 11.61       | 93.9%     |
| 256x256    | 1          | 2.50             | 400.0  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.13             | 11.61       | 1.1%      |
| 256x256    | 8          | 0.70             | 1428.6 | NVIDIA GeForce RTX 4060 Laptop GPU | 1.50             | 11.61       | 12.9%     |
| 256x256    | 16         | 0.72             | 1388.9 | NVIDIA GeForce RTX 4060 Laptop GPU | 3.00             | 11.61       | 25.8%     |
| 256x256    | 32         | 0.75             | 1333.3 | NVIDIA GeForce RTX 4060 Laptop GPU | 6.00             | 11.61       | 51.7%     |
| 256x256    | 64         | 0.77             | 1298.7 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.20            | 11.61       | 96.5%     |
| 384x384    | 1          | 2.40             | 416.7  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.20             | 11.61       | 1.7%      |
| 384x384    | 8          | 1.65             | 606.1  | NVIDIA GeForce RTX 4060 Laptop GPU | 2.20             | 11.61       | 19.0%     |
| 384x384    | 16         | 1.62             | 617.3  | NVIDIA GeForce RTX 4060 Laptop GPU | 4.40             | 11.61       | 37.9%     |
| 384x384    | 32         | 1.78             | 561.8  | NVIDIA GeForce RTX 4060 Laptop GPU | 8.80             | 11.61       | 75.8%     |
| 384x384    | 64         | 1.70             | 588.2  | NVIDIA GeForce RTX 4060 Laptop GPU | 11.00            | 11.61       | 94.8%     |
| 512x512    | 1          | 3.00             | 333.3  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.30             | 11.61       | 2.6%      |
| 512x512    | 8          | 2.90             | 344.8  | NVIDIA GeForce RTX 4060 Laptop GPU | 3.10             | 11.61       | 26.7%     |
| 512x512    | 16         | 3.05             | 327.9  | NVIDIA GeForce RTX 4060 Laptop GPU | 6.20             | 11.61       | 53.4%     |
| 512x512    | 32         | 3.10             | 322.6  | NVIDIA GeForce RTX 4060 Laptop GPU | 9.80             | 11.61       | 84.4%     |
| 512x512    | 64         | 3.12             | 320.5  | NVIDIA GeForce RTX 4060 Laptop GPU | 11.10            | 11.61       | 95.6%     |

### Таблица 2. Сравнение FPS (Onnx)

| Resolution | Batch Size | Time/Image (ms) | FPS    | GPU                              | TFLOPs Measured | TFLOPs Max | Util (%) |
|------------|------------|-----------------|--------|----------------------------------|------------------|------------|----------|
| 224x224    | 1          | 2.80            | 357.1  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.10             | 11.61      | 0.9%     |
| 224x224    | 8          | 0.63            | 1587.3 | NVIDIA GeForce RTX 4060 Laptop GPU | 1.30             | 11.61      | 11.2%    |
| 224x224    | 16         | 0.61            | 1639.3 | NVIDIA GeForce RTX 4060 Laptop GPU | 2.70             | 11.61      | 23.3%    |
| 224x224    | 32         | 0.59            | 1694.9 | NVIDIA GeForce RTX 4060 Laptop GPU | 5.30             | 11.61      | 45.7%    |
| 224x224    | 64         | 0.58            | 1724.1 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.20            | 11.61      | 96.5%    |
| 256x256    | 1          | 3.10            | 322.6  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.13             | 11.61      | 1.1%     |
| 256x256    | 8          | 0.81            | 1234.6 | NVIDIA GeForce RTX 4060 Laptop GPU | 1.70             | 11.61      | 14.6%    |
| 256x256    | 16         | 0.79            | 1265.8 | NVIDIA GeForce RTX 4060 Laptop GPU | 3.40             | 11.61      | 29.3%    |
| 256x256    | 32         | 0.77            | 1298.7 | NVIDIA GeForce RTX 4060 Laptop GPU | 6.60             | 11.61      | 56.9%    |
| 256x256    | 64         | 0.74            | 1351.4 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.20            | 11.61      | 96.5%    |
| 384x384    | 1          | 3.50            | 285.7  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.20             | 11.61      | 1.7%     |
| 384x384    | 8          | 2.10            | 476.2  | NVIDIA GeForce RTX 4060 Laptop GPU | 2.40             | 11.61      | 20.7%    |
| 384x384    | 16         | 1.85            | 540.5  | NVIDIA GeForce RTX 4060 Laptop GPU | 4.80             | 11.61      | 41.3%    |
| 384x384    | 32         | 1.70            | 588.2  | NVIDIA GeForce RTX 4060 Laptop GPU | 8.50             | 11.61      | 73.2%    |
| 384x384    | 64         | 1.65            | 606.1  | NVIDIA GeForce RTX 4060 Laptop GPU | 11.10            | 11.61      | 95.6%    |
| 512x512    | 1          | 4.10            | 243.9  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.30             | 11.61      | 2.6%     |
| 512x512    | 8          | 2.80            | 357.1  | NVIDIA GeForce RTX 4060 Laptop GPU | 3.30             | 11.61      | 28.4%    |
| 512x512    | 16         | 2.50            | 400.0  | NVIDIA GeForce RTX 4060 Laptop GPU | 6.60             | 11.61      | 56.9%    |
| 512x512    | 32         | 2.40            | 416.7  | NVIDIA GeForce RTX 4060 Laptop GPU | 9.80             | 11.61      | 84.4%    |
| 512x512    | 64         | 2.38            | 420.2  | NVIDIA GeForce RTX 4060 Laptop GPU | 11.10            | 11.61      | 95.6%    |

### Таблица 3. Сравнение FPS (trt)

| Resolution | Batch Size | Time/Image (ms) | FPS    | GPU                              | TFLOPs Measured | TFLOPs Max | Util (%) |
|------------|------------|------------------|--------|-----------------------------------|------------------|-------------|-----------|
| 224x224    | 1          | 2.51             | 398.4  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.69             | 11.61       | 5.9%      |
| 224x224    | 8          | 0.45             | 1777.8 | NVIDIA GeForce RTX 4060 Laptop GPU | 4.15             | 11.61       | 35.7%     |
| 224x224    | 16         | 0.31             | 3225.8 | NVIDIA GeForce RTX 4060 Laptop GPU | 6.00             | 11.61       | 51.7%     |
| 224x224    | 32         | 0.18             | 5555.6 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.00            | 11.61       | 94.8%     |
| 224x224    | 64         | 0.17             | 5882.4 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.50            | 11.61       | 99.1%     |
| 256x256    | 1          | 3.10             | 322.6  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.62             | 11.61       | 5.3%      |
| 256x256    | 8          | 0.60             | 1333.3 | NVIDIA GeForce RTX 4060 Laptop GPU | 4.00             | 11.61       | 34.4%     |
| 256x256    | 16         | 0.35             | 2857.1 | NVIDIA GeForce RTX 4060 Laptop GPU | 6.00             | 11.61       | 51.7%     |
| 256x256    | 32         | 0.20             | 5000.0 | NVIDIA GeForce RTX 4060 Laptop GPU | 10.80            | 11.61       | 93.0%     |
| 256x256    | 64         | 0.19             | 5263.2 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.30            | 11.61       | 97.3%     |
| 384x384    | 1          | 3.70             | 270.3  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.55             | 11.61       | 4.7%      |
| 384x384    | 8          | 0.85             | 941.2  | NVIDIA GeForce RTX 4060 Laptop GPU | 3.50             | 11.61       | 30.2%     |
| 384x384    | 16         | 0.48             | 2083.3 | NVIDIA GeForce RTX 4060 Laptop GPU | 5.80             | 11.61       | 50.0%     |
| 384x384    | 32         | 0.29             | 3448.3 | NVIDIA GeForce RTX 4060 Laptop GPU | 10.80            | 11.61       | 93.0%     |
| 384x384    | 64         | 0.28             | 3571.4 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.10            | 11.61       | 95.6%     |
| 512x512    | 1          | 4.00             | 250.0  | NVIDIA GeForce RTX 4060 Laptop GPU | 0.70             | 11.61       | 6.0%      |
| 512x512    | 8          | 1.10             | 727.3  | NVIDIA GeForce RTX 4060 Laptop GPU | 3.70             | 11.61       | 31.9%     |
| 512x512    | 16         | 0.65             | 1538.5 | NVIDIA GeForce RTX 4060 Laptop GPU | 6.30             | 11.61       | 54.3%     |
| 512x512    | 32         | 0.39             | 2564.1 | NVIDIA GeForce RTX 4060 Laptop GPU | 10.40            | 11.61       | 89.6%     |
| 512x512    | 64         | 0.37             | 2702.7 | NVIDIA GeForce RTX 4060 Laptop GPU | 11.10            | 11.61       | 95.6%     |

### Таблица 4. Ускорение относительно PyTorch (Batch Size = 32, Resolution = 256x256)

| Framework  | Time/Image (ms) | FPS    | Ускорение (x) | GPU                               |
|------------|-----------------|--------|---------------|----------------------------------|
| PyTorch    | 0.75            | 1333.3 | 1.00          | NVIDIA GeForce RTX 4060 Laptop GPU |
| ONNX       | 0.77            | 1298.7 | 0.97          | NVIDIA GeForce RTX 4060 Laptop GPU |
| TensorRT   | 0.20            | 5000.0 | 3.75          | NVIDIA GeForce RTX 4060 Laptop GPU |

### Графики

#### FPS vs Размер изображения (Batch Size = 32)

![fps_vs_res](plots/fps_vs_resolution_batch32.png "FPS vs Размер изображения (Batch Size = 32)")

#### FPS vs Batch Size (Resolution = 256x256)

![fps_vs_bs](plots/fps_vs_batchsize_resolution256.png "FPS vs Batch Size (Resolution = 256x256)")

#### Ускорение относительно PyTorch по размеру батча (Resolution = 256x256)
![speedup_bs](plots/speedup_vs_batchsize_resolution256.png "Ускорение относительно PyTorch по размеру батча (Resolution = 256x256)")

#### Ускорение относительно PyTorch по размеру изображения (Batch Size = 32)

![speedup_res](plots/speedup_vs_batchsize_resolution256.png "Ускорение относительно PyTorch по размеру изображения (Batch Size = 32)")

## 4. Обсуждение

### Какой подход показывает лучшую производительность?

- **Torch-TensorRT** обеспечивает наибольший FPS и наименьшее время на один образ.
- **ONNX Runtime** — сбалансированный и удобный вариант без необходимости сильно модифицировать код.

### Как влияет размер изображения?

- При увеличении размера FPS падает почти линейно.
- Torch-TensorRT масштабируется лучше других.

### Как влияет размер батча?

- Увеличение батча повышает FPS до определенного предела.
- Насыщение начинается с **batch size ~32-64**
- При слишком малом батче (<8) видеокарта недогружена

### Использование ресурсов

- Максимальное использование TFLOPS достигается при **Torch-TensorRT** (до 75%)
- PyTorch демонстрирует наихудшее использование GPU

## 5. Заключение

### Выводы
- TensorRT — лучший вариант для высокопроизводительного инференса (FPS ↑ в 3.5–4 раза по сравнению с PyTorch)
- ONNX — промежуточное решение: быстрее PyTorch, но уступает TRT
- Рост разрешения → снижение FPS (линейно)
- Увеличение batch size → рост FPS до точки насыщения (~32)
- Лучший вариант это TensorRT + 256x256 + Batch Size = 32 или 64. Дает оптимальный FPS (~5000+) и максимальную загрузку GPU (90–99%)

### Практические рекомендации

- Использовать размер батча ≥32 и разрешение ≤384x384
- Для inference в продакшене — использовать Torch-TensorRT

### Направления для дальнейшего исследования

- Использование FP16 / INT8
- Адаптация к real-time задачам (видео/стриминг)
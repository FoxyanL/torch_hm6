import os
from typing import Tuple

import torch
import torch_tensorrt as t_trt
from torch.utils.data import Dataset

try:
    from core.model import Resnet18
    from core.datasets import CustomImageDataset, RandomImageDataset
    from core.utils import run_test
except ImportError:
    from core.model import Resnet18
    from core.datasets import CustomImageDataset, RandomImageDataset
    from core.utils import run_test


def convert_to_torch_trt(
    model_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'fp16',
    workspace_size: int = 1 << 30,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    opt_batch_size: int = 1,
    **kwargs
):
    """
    Конвертирует PyTorch модель в TensorRT через torch-tensorrt
    
    Args:
        model_path: Путь к сохраненной PyTorch модели (.pth)
        input_shape: Форма входного тензора (C, H, W)
        precision: Точность (fp32 или fp16)
        workspace_size: Размер рабочего пространства (байты)
        min_batch_size: Минимальный размер батча (для динамического батча)
        max_batch_size: Максимальный размер батча
        opt_batch_size: Оптимальный размер батча
    
    Returns:
        trt_model: скомпилированная TensorRT модель
    """
    # Загружаем модель
    model = Resnet18()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to('cuda')
    model.eval()
    
    # Создаем директорию для сохранения если нужно
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    precision_dtype = torch.float16 if precision == 'fp16' else torch.float32
    min_shape = (min_batch_size, *input_shape)
    opt_shape = (opt_batch_size, *input_shape)
    max_shape = (max_batch_size, *input_shape)
    
    # Компилируем модель в TensorRT
    trt_model = t_trt.compile(
        model,
        inputs=[t_trt.Input(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            dtype=precision_dtype
        )],
        enabled_precisions={precision_dtype},
        workspace_size=workspace_size
    )

    # Создаем фиктивные тензоры для сохранения модели
    inputs = [
        torch.randn(min_shape, device='cuda', dtype=precision_dtype),
        torch.randn(opt_shape, device='cuda', dtype=precision_dtype),
        torch.randn(max_shape, device='cuda', dtype=precision_dtype)
    ]
    
    # Путь для сохранения TRT модели
    output_path = model_path.replace('.pth', '.trt')
    t_trt.save(trt_model, output_path, inputs=inputs)
    
    print(f"Модель успешно конвертирована в TensorRT (torch-tensorrt): {output_path}")
    return trt_model


def test_torch_trt_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataset: Dataset = None,
    **kwargs
) -> dict[Tuple[int, int, int], float]:
    """
    Тестирует производительность torch-tensorrt модели
    
    Args:
        model: torch_tensorrt скомпилированная модель
        input_shape: форма входных данных (C, H, W)
        num_runs: количество повторов для измерения времени
        min_batch_size: минимальный размер батча
        max_batch_size: максимальный размер батча
        batch_step: шаг увеличения батча
        dataset: опциональный датасет для теста
    
    Returns:
        dict с размерами батча и средним временем на изображение (мс)
    """
    return run_test(
        model_wrapper=model,
        input_shape=input_shape,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_step=batch_step,
        dataset=dataset,
        timer_type='cuda'
    )

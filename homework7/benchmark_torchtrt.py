import torch
import torch_tensorrt as t_trt
import time
import os
from typing import Tuple, List

from export_trt import convert_to_torch_trt

MAX_TFLOPS = 11.61
GPU_NAME = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"

def measure_inference_time(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100
) -> float:
    """
    Измеряет среднее время инференса модели на одном батче.

    Args:
        model (torch.nn.Module): Модель Torch или Torch-TensorRT.
        input_tensor (torch.Tensor): Входной тензор с размерами (batch_size, C, H, W).
        num_runs (int): Количество итераций инференса для усреднения.

    Returns:
        float: Среднее время инференса одного батча, приведенное к времени на одно изображение (в миллисекундах).
    """
    model.eval()
    model.to('cuda')
    
    # берем dtype из input_tensor
    input_tensor = input_tensor.to('cuda').to(input_tensor.dtype)

    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = end - start
    avg_time_ms = (total_time / num_runs) * 1000
    return avg_time_ms



def calculate_tflops_measured(batch_size: int, img_size: int, time_per_image_ms: float) -> float:
    """
    Оценивает измеренную производительность модели в TFLOPs.

    Args:
        batch_size (int): Размер батча.
        img_size (int): Размер стороны квадратного изображения.
        time_per_image_ms (float): Среднее время инференса одного изображения (в мс).

    Returns:
        float: Оценка производительности модели в терафлопсах (TFLOPs).
    """
    flops_per_image = 1.8 * 10**9
    ops_per_sec = (batch_size * flops_per_image) / (time_per_image_ms / 1000)
    tflops = ops_per_sec / 1e12
    return tflops

def run_benchmark(
    model_path: str,
    output_md_path: str = "benchmark_results.md",
    image_sizes: List[int] = [224, 256, 384, 512],
    batch_sizes: List[int] = [1, 8, 16, 32, 64],
    precision: str = 'fp16',
):
    """
    Проводит бенчмарк Torch-TensorRT модели на разных размерах изображений и батчей.

    Загружает модель, конвертирует её в Torch-TensorRT, измеряет время инференса,
    оценивает TFLOPs и сохраняет результаты в markdown-таблицу.

    Args:
        model_path (str): Путь к файлу с весами модели (.pth).
        output_md_path (str): Путь для сохранения markdown-файла с результатами.
        image_sizes (List[int]): Список размеров входных изображений (высота и ширина одинаковы).
        batch_sizes (List[int]): Список размеров батча для инференса.
        precision (str): Точность модели Torch-TensorRT (например, 'fp16' или 'fp32').

    Returns:
        None
    """
    lines = []
    lines.append(f"GPU: {GPU_NAME}\n")
    lines.append("| Resolution | Batch Size | Time/Image (ms) | GPU | TFLOPs Measured | TFLOPs Max | Util (%) |")
    lines.append("|------------|------------|-----------------|-----|-----------------|------------|----------|")
    
    for image_size in image_sizes:
        input_shape = (3, image_size, image_size)
        trt_model = convert_to_torch_trt(
            model_path=model_path,
            input_shape=input_shape,
            precision=precision,
            workspace_size=1 << 30,
            min_batch_size=1,
            opt_batch_size=max(batch_sizes),
            max_batch_size=max(batch_sizes)
        )
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, *input_shape, dtype=torch.float16).cuda()
            time_per_img = measure_inference_time(trt_model, input_tensor, num_runs=100)
            tflops_measured = calculate_tflops_measured(batch_size, image_size, time_per_img)
            util_percent = (tflops_measured / MAX_TFLOPS) * 100
            
            line = f"| {image_size}x{image_size} | {batch_size} | {time_per_img:.2f} | {GPU_NAME} | {tflops_measured:.2f} | {MAX_TFLOPS:.2f} | {util_percent:.1f}% |"
            lines.append(line)
            print(line)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\nРезультаты сохранены в файл: {output_md_path}")

if __name__ == "__main__":
    model_path = "weights/best_resnet18.pth"
    run_benchmark(model_path)

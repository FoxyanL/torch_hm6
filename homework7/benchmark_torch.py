import torch
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.model import Resnet18
from core.datasets import RandomImageDataset
from ptflops import get_model_complexity_info

def get_gpu_name():
    """
    Возвращает имя доступной видеокарты CUDA.

    Returns:
        str: Название GPU, если CUDA доступна, иначе "CPU".
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def estimate_flops(model, input_size, batch_size):
    """
    Оценивает количество FLOPs (Floating Point Operations) для одного батча.

    Args:
        model (torch.nn.Module): PyTorch-модель.
        input_size (tuple): Размер входного изображения (C, H, W).
        batch_size (int): Размер батча.

    Returns:
        float: Оценка количества FLOPs на один батч, в терафлопсах (TFLOPs).
    """
    macs, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    flops = macs * 2  # MACs -> FLOPs
    return flops * batch_size / 1e12  # TFLOPs per batch

def benchmark(model, dataloader, device):
    """
    Измеряет среднее время инференса одного изображения на заданной модели.

    Args:
        model (torch.nn.Module): PyTorch-модель для тестирования.
        dataloader (DataLoader): DataLoader с изображениями.
        device (torch.device): Устройство для выполнения инференса (CPU или CUDA).

    Returns:
        float: Среднее время инференса одного изображения (в секундах).
    """
    model.eval().to(device)
    times = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Benchmarking", leave=False):
            images = images.to(device)
            torch.cuda.synchronize()
            start = time.time()
            _ = model(images)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) / images.shape[0])
    return sum(times) / len(times)  # avg sec per image

def save_results_markdown(results, filename="pytorch_benchmark_report.md"):
    """
    Сохраняет результаты бенчмарка в markdown-таблицу.

    Args:
        results (list): Список кортежей с результатами бенчмарка.
        filename (str): Имя файла для сохранения markdown-таблицы.

    Returns:
        None
    """
    with open(filename, "w") as f:
        f.write("| Resolution | Batch Size | Time/Image (ms) | FPS | GPU | TFLOPs Measured | TFLOPs Max | Util (%) |\n")
        f.write("|------------|------------|------------------|------|----------------|------------------|----------------|-----------|\n")
        for row in results:
            res, batch, t, fps, gpu, flops, max_flops, util = row
            f.write(f"| {res}x{res} | {batch} | {t * 1000:.2f} | {fps:.1f} | {gpu} | {flops:.2f} | {max_flops:.2f} | {util:.1f}% |\n")
    print(f"Markdown-таблица сохранена в {filename}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolutions = [224]
    batch_sizes = [1, 8, 16, 32, 64]
    gpu_name = get_gpu_name()
    theoretical_tflops = 11.61

    results = []

    for res in resolutions:
        for batch in batch_sizes:
            print(f"{res}x{res}, batch={batch}")
            model = Resnet18(num_classes=1000)
            input_shape = (3, res, res)
            dataset = RandomImageDataset(target_size=input_shape)
            dataloader = DataLoader(dataset, batch_size=batch, drop_last=True)

            time_per_img = benchmark(model, dataloader, device)  # sec
            fps = 1 / time_per_img
            measured_tflops = estimate_flops(model, input_shape, batch)
            utilization = (measured_tflops / theoretical_tflops) * 100

            results.append((res, batch, time_per_img, fps, gpu_name, measured_tflops, theoretical_tflops, utilization))

    save_results_markdown(results)

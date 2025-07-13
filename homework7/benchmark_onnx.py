import onnxruntime as ort
import numpy as np
import torch
import time
import platform
import GPUtil
import os


OUTPUT_PATH = "onnx_benchmark.md"
TFLOPS_MAX = 11.61
RESOLUTIONS = [224, 256, 384, 512]
BATCH_SIZES = [1, 8, 16, 32, 64]

def get_gpu_name():
    """
    Возвращает название первого доступного GPU с помощью библиотеки GPUtil.

    Returns:
        str: Название GPU или "CPU", если GPU не обнаружен.
    """
    gpus = GPUtil.getGPUs()
    return gpus[0].name if gpus else "CPU"

def benchmark_onnx_model(onnx_path, image_size=224, batch_size=32, warmup=10, runs=50):
    """
    Выполняет бенчмарк ONNX-модели, измеряя среднее время инференса и вычислительную загрузку GPU.

    Args:
        onnx_path (str): Путь к файлу ONNX модели.
        image_size (int, optional): Размер стороны квадратного входного изображения. По умолчанию 224.
        batch_size (int, optional): Размер батча для инференса. По умолчанию 32.
        warmup (int, optional): Количество прогревочных запусков модели. По умолчанию 10.
        runs (int, optional): Количество замеряемых прогонов модели. По умолчанию 50.

    Returns:
        dict: Словарь с результатами, содержащий:
            - "Resolution" (str): Размер изображения, например, "224x224".
            - "Batch Size" (int): Размер батча.
            - "Time/Image (ms)" (str): Среднее время обработки одного изображения в миллисекундах.
            - "GPU" (str): Название GPU.
            - "TFLOPs Measured" (str): Измеренная производительность в терафлопсах.
            - "TFLOPs Max" (str): Теоретический максимум TFLOPs.
            - "Util (%)" (str): Процент загрузки GPU.
    """
    ort_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    
    input_shape = (batch_size, 3, image_size, image_size)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up
    for _ in range(warmup):
        _ = ort_session.run(None, {"input": dummy_input})
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        _ = ort_session.run(None, {"input": dummy_input})
    torch.cuda.synchronize()
    total_time = time.time() - start

    avg_time_per_batch = total_time / runs * 1000  # ms
    avg_time_per_image = avg_time_per_batch / batch_size

    # FLOPs
    flops_per_image = 1.8e9
    tflops = (flops_per_image * batch_size) / (avg_time_per_batch / 1000) / 1e12
    utilization = (tflops / TFLOPS_MAX) * 100

    return {
        "Resolution": f"{image_size}x{image_size}",
        "Batch Size": batch_size,
        "Time/Image (ms)": f"{avg_time_per_image:.2f}",
        "GPU": get_gpu_name(),
        "TFLOPs Measured": f"{tflops:.2f}",
        "TFLOPs Max": f"{TFLOPS_MAX:.2f}",
        "Util (%)": f"{utilization:.1f}%"
    }

def save_markdown_table(results, output_path):
    """
    Сохраняет список результатов бенчмарка в markdown-файл в виде таблицы.

    Args:
        results (list): Список словарей с результатами бенчмарка.
        output_path (str): Путь к выходному markdown-файлу.

    Returns:
        None
    """
    header = "| Resolution | Batch Size | Time/Image (ms) | GPU | TFLOPs Measured | TFLOPs Max | Util (%) |\n"
    separator = "|------------|------------|------------------|----------------|------------------|----------------|-----------|\n"
    
    with open(output_path, "w") as f:
        f.write(header)
        f.write(separator)
        for row in results:
            f.write(f"| {row['Resolution']} | {row['Batch Size']} | {row['Time/Image (ms)']} | {row['GPU']} | "
                    f"{row['TFLOPs Measured']} | {row['TFLOPs Max']} | {row['Util (%)']} |\n")

if __name__ == "__main__":
    all_results = []

    for image_size in RESOLUTIONS:
        for batch_size in BATCH_SIZES:
            onnx_path = f"weights/resnet18_{image_size}.onnx"
            result = benchmark_onnx_model(onnx_path, image_size=image_size, batch_size=batch_size)
            all_results.append(result)
            print(f"{result['Resolution']} | Batch {batch_size} done")

    save_markdown_table(all_results, OUTPUT_PATH)
    print(f"Результаты сохранены в {OUTPUT_PATH}")

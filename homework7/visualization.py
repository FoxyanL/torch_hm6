import os
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

table_torch = """
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

"""

table_onnx = """
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

"""
table_trt = """
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

"""

def load_table(table_md: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(table_md.strip()), sep="|", skipinitialspace=True)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df.columns = [col.strip() for col in df.columns]
    df = df[~df['Resolution'].str.contains('---')]
    return df

df_torch = load_table(table_torch)
df_onnx = load_table(table_onnx)
df_trt = load_table(table_trt)


def extract_resolution(res_str: str) -> int:
    """'224x224' -> 224"""
    return int(res_str.split('x')[0])

for df in [df_torch, df_onnx, df_trt]:
    df['Resolution_int'] = df['Resolution'].apply(extract_resolution)
    df['Batch Size'] = df['Batch Size'].astype(int)
    df['FPS'] = df['FPS'].astype(float)


os.makedirs("plots", exist_ok=True)

# FPS vs Размер изображения (batch size = 32)
plt.figure(figsize=(8,6))
batch_fixed = 32
for df, label, color in zip([df_torch, df_onnx, df_trt], ["PyTorch", "ONNX", "Torch-TensorRT"], ["blue", "green", "red"]):
    df_sub = df[df["Batch Size"] == batch_fixed]
    plt.plot(df_sub['Resolution_int'], df_sub['FPS'], marker='o', label=label, color=color)
plt.title(f"FPS vs Размер изображения (батч = {batch_fixed})")
plt.xlabel("Размер изображения")
plt.ylabel("FPS")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/fps_vs_resolution_batch{batch_fixed}.png")
plt.close()

# FPS vs Размер батча (фиксируем resolution = 256)
plt.figure(figsize=(8,6))
resolution_fixed = 256
for df, label, color in zip([df_torch, df_onnx, df_trt], ["PyTorch", "ONNX", "Torch-TensorRT"], ["blue", "green", "red"]):
    df_sub = df[df["Resolution_int"] == resolution_fixed]
    plt.plot(df_sub['Batch Size'], df_sub['FPS'], marker='o', label=label, color=color)
plt.title(f"FPS vs Размер батча (размер изображения = {resolution_fixed})")
plt.xlabel("Размер батча")
plt.ylabel("FPS")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/fps_vs_batchsize_resolution{resolution_fixed}.png")
plt.close()

# Ускорение относительно PyTorch

# вычисление ускорения по FPS: fps_метод / fps_pytorch
def calc_speedup(df_method, df_ref):
    """вычисление ускорения по FPS: fps_метод / fps_pytorch"""
    merged = pd.merge(df_method, df_ref, on=['Resolution', 'Batch Size'], suffixes=('', '_ref'))
    merged['Speedup'] = merged['FPS'] / merged['FPS_ref']
    return merged

# ускорение относительно PyTorch
speedup_onnx = calc_speedup(df_onnx, df_torch)
speedup_trt = calc_speedup(df_trt, df_torch)

# ускорение по размеру изображения при batch_size=32
plt.figure(figsize=(8,6))
batch_fixed = 32
for speedup_df, label, color in zip([speedup_onnx, speedup_trt], ["ONNX", "Torch-TensorRT"], ["green", "red"]):
    df_sub = speedup_df[speedup_df['Batch Size'] == batch_fixed]
    plt.plot(df_sub['Resolution_int'], df_sub['Speedup'], marker='o', label=label, color=color)
plt.axhline(1, color='gray', linestyle='--')
plt.title(f"Ускорение относительно PyTorch (батч = {batch_fixed}) по размеру изображения")
plt.xlabel("Размер изображения")
plt.ylabel("Ускорение")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/speedup_vs_resolution_batch{batch_fixed}.png")
plt.close()

# ускорение по размеру батча при resolution=256
plt.figure(figsize=(8,6))
resolution_fixed = 256
for speedup_df, label, color in zip([speedup_onnx, speedup_trt], ["ONNX", "Torch-TensorRT"], ["green", "red"]):
    df_sub = speedup_df[speedup_df['Resolution_int'] == resolution_fixed]
    plt.plot(df_sub['Batch Size'], df_sub['Speedup'], marker='o', label=label, color=color)
plt.axhline(1, color='gray', linestyle='--')
plt.title(f"Ускорение относительно PyTorch (размер изображения = {resolution_fixed}) по размеру батча")
plt.xlabel("Размер батча")
plt.ylabel("Ускорение")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/speedup_vs_batchsize_resolution{resolution_fixed}.png")
plt.close()

print("Графики сохранены в папку plots/")

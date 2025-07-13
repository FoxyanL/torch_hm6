import torch
from core.model import Resnet18
import os

resolutions = [224, 256, 384, 512]
model_path = "weights/best_resnet18.pth"
save_dir = "weights"

os.makedirs(save_dir, exist_ok=True)

# Загрузка модели один раз
model = Resnet18(num_classes=1000)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Экспортируем для всех разрешений
for image_size in resolutions:
    dummy_input = torch.randn(1, 3, image_size, image_size)
    onnx_path = os.path.join(save_dir, f"resnet18_{image_size}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"ONNX-модель сохранена: {onnx_path}")

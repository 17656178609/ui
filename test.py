import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.current_device())  # 输出 GPU 设备索引（如 0）
print(torch.version.cuda)          # 输出 CUDA 版本（如 11.8）

import torch
print(torch.__version__)  # 应输出 PyTorch 版本
print(torch.cuda.is_available())  # 应返回 True
print(torch.cuda.get_device_name(0))  # 打印 GPU 名称
import torch

# 第一关：装上了吗、什么版本
# 打印 torch.__version__ 和 torch.version.cuda
#   → 期望：2.11.0+cu128 / 12.8

print(torch.__version__)    # 2.11.0+cu128
print(torch.version.cuda)   # 12.8

# 第二关：显卡认出来了吗
# 打印 torch.cuda.is_available()
#      torch.cuda.get_device_name(0)
#      torch.cuda.get_device_capability(0)
#   → 期望：True / NVIDIA GeForce RTX 5060 Laptop GPU / (12, 0)

print(torch.cuda.is_available())                # True
print(torch.cuda.get_device_name(0))            # NVIDIA GeForce RTX 5060 Laptop GPU
print(torch.cuda.get_device_capability(0))      # (12, 0)

# 第三关（最重要，别跳过）：真的能算吗
# 造两个 1000x1000 的随机张量，放到 cuda 上，做矩阵乘法，打印结果的和
#   x = torch.randn(1000, 1000, device="cuda")
#   y = x @ x
#   print(y.sum().item())        ← 这一行会真的调用 GPU 内核

x = torch.randn(1000, 1000, device="cuda")
y = x @ x
print(y.sum().item())

# 第四关：把张量搬来搬去
# 把 x 搬到 cpu，打印 x.device；再搬回 cuda，打印 x.device

x = torch.randn(1000, 1000, device="cuda")  # 造一个，在 cuda
print(x.device)            # cuda:0
y = x.to("cpu")            # ← 搬家：把 x 搬到 cpu
print(x.device, y.device)  # cuda:0 cpu   ← 两个都打印

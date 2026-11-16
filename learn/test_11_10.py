import torch
import numpy as np

# 1. 创建一个 2行3列 的 2维张量 (矩阵)
t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6]])

# 不加括号代表它是对象“自带的静态特征”，加括号代表它是对象“需要执行的动作或计算”，张量在底层只存了形状
print("张量内容:\n", t)
print("张量的形状 (shape):", t.shape)         # torch.Size([2, 3])
print("张量中元素的总数 (numel):", t.numel()) # 6                           NumPy 里获取元素总数的属性叫 .size
print("张量的维度数 (dim):", t.dim())         # 2
print("张量的数据类型 (dtype):", t.dtype)     # torch.int64

# 2. 张量的超能力演示：移动到 GPU
if torch.cuda.is_available():
    t_gpu = t.to('cuda')  # 将张量复制到显存中
    print("张量已成功移动到:", t_gpu.device) # 输出: cuda:0

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X == Y)
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

import numpy as np
import torch

# 1) 同一个东西，两种身份
#    用 numpy 造 arr = np.arange(12).reshape(3, 4)
#    用 torch 造 t  = torch.arange(12).reshape(3, 4)
#    分别打印 type / shape / dtype
#    回答：dtype 一样吗？（numpy 默认 int64，torch 默认 int64 还是 int32？）

arr = np.arange(12).reshape(3, 4)
t = torch.arange(12).reshape(3, 4) # .reshape((3, 4))其实构造的是3维张量
print(f"数组的类型是：{type(arr)}，张量的类型是：{t.type()}")
print(f"数组的元素类型是：{arr.dtype}，张量的元素类型是：{t.dtype}")
print(f"数组的形状是：{arr.shape}，张量的形状是：{t.shape}")

# 2) 五种创建方式，各造一个 2x3
#    torch.tensor([[1,2,3],[4,5,6]])
#    torch.zeros((2,3))   torch.ones((2,3))
#    torch.arange(6).reshape(2,3)
#    torch.randn(2,3)     ← 标准正态分布，W6 造假数据要用
#    打印各自的 dtype，看 zeros/ones 是 float 还是 int

t_1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
t_2 = torch.ones((2, 3))
t_3 = torch.zeros((2, 3))
t_4 = torch.arange(6).reshape(2, 3)
t_5 = torch.randn(2, 3)  # # 2=行数, 3=列数；randn 固定 N(0,1)，改不了

print(t_1.dtype) # torch.int64
print(t_2.dtype) # torch.float32
print(t_3.dtype) # torch.float32
print(t_4.dtype) # torch.int64
print(t_5.dtype) # torch.float32

# 3) 四件套：data / shape / dtype / device
#    对第 2 题的某一个张量，打印 shape、dtype、device
#    device 打印出来是什么？（期望 cpu）

print(t_1.device)  # cpu
print(t_1.shape)   # torch.Size([2, 3])
print(t_1.dtype)   # torch.int64

# 4) 搬到显卡上
#    t = torch.randn(2, 3)
#    t_gpu = t.to("cuda")        ← 也可以写 t.cuda()
#    打印 t.device 和 t_gpu.device
#    再 t_gpu.to("cpu")，打印 device
#    回答：t 和 t_gpu 是同一个东西吗？（用 data_ptr() 对比一下）

t = torch.randn(2, 3)
t_gpu = t.to("cuda")
print(t.device, t_gpu.device)          # cpu cuda:0
print(t.data_ptr(), t_gpu.data_ptr())  # 5279790072192 47343206400  没有装在同一块内存里

# 5) 和 numpy 互转，验证"共享内存"
#    a = np.array([1, 2, 3])
#    b = torch.from_numpy(a)
#    b[0] = 999
#    print(a)                    ← 看 a 变没变
#    回答：这和你 W4 D1 验证过的"reshape 不搬数据"是不是同一件事？ ———— 应该是同一件事

"""
	        是不是同一个对象	            共不共用内存
reshape	    是，同一个数组换形状看	             是
from_numpy	否，一个 numpy 一个 torch	        是
"""

a = np.array([1, 2, 3])
b = torch.from_numpy(a)
b[0] = 999
print(a)  # [999   2   3]    生成的 PyTorch 张量和原来的 NumPy 数组，在底层指向的是同一块物理内存

# 6) 反向：张量转 numpy
#    c = torch.tensor([1.0, 2.0])
#    c.numpy()                   ← 能转吗？
#    那如果 c 已经在 cuda 上呢？试一下 t_gpu.numpy()，看它报什么

c = torch.tensor([1.0, 2.0])
c_arr_1 = c.numpy()
print(c_arr_1)
# 直接 c.to("cuda") 相当于把返回值给扔了
c = c.to("cuda") 
# c_arr_2 = c.numpy()  # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
# print(c_arr_2)  
# print(t_gpu.numpy())  
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.


x = torch.randn(2, 3)
print(x.shape)                  # torch.Size([2, 3])
print(x.mean())                 # tensor(0.1815)
print(x.std())                  # tensor(0.4669)
print(torch.randn(0, 1).shape)  # torch.Size([0, 1])
print(torch.randn(0, 1))        # tensor([], size=(0, 1))
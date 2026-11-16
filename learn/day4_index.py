import torch
import numpy as np
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
from week3_csv_stats import read_csv, clean

t = torch.arange(24).reshape(4, 6)

# 1) 和 numpy 逐条对照（每条后面写注释："和 numpy 一样" 或 "不一样"）
#    t[0]        t[0, 0]
#    t[:, 1]     t[1, :]
#    t[1:3, :2]
#    每一条都打印 shape，和 W4 D2 的结果对一遍

print(t[0], t[0].shape)                # 一样
print(t[0, 0], t[0, 0].shape)          # 一样
print(t[:, 1], t[:, 1].shape)          # 一样
print(t[1, :], t[1, :].shape)          # 一样
print(t[1:3, :2], t[1:3, :2].shape)    # 一样
print()

# 2) 布尔索引
#    t[t > 10]              → 打印结果和 shape
#    t[(t > 5) & (t < 10)]  → 注意括号和 &
#    t[(t < 5) | (t > 20)]
#    回答：和 W4 的 arr[arr > 30] 是一回事吗？ ————— 是一回事

print(t[t>10], t[t>10].shape)
print(t[(t>5) & (t<10)], t[(t>5) & (t<10)].shape)
print(t[(t<5) | (t>20)], t[(t<5) | (t>20)].shape)
print()

# 3) ⭐ 新东西：.item()
#    s = t.sum()            → 打印它，看是什么类型
#    print(type(s))         ← 它是 tensor，不是数字
#    print(s.item())        ← 这才是 Python 的数字
#    回答：为什么要专门搞一个 .item()？ 

print(torch.tensor(3.0).item())         # 3.0
print(torch.tensor([3.0]).item())       # 3.0   箱子只有一个苹果，也取得出来
# print(torch.tensor([3.0, 4.0]).item())  # 报错！


s = t.sum()
print(type(s))  # <class 'torch.Tensor'>
print(s.item()) # 276
print()

# 4) 单元素张量
#    x = torch.tensor([3.0])
#    print(x, x.shape, x.item())
#    print(torch.tensor(3.0).shape)    ← 注意 0 维
#    回答：shape 是 () 的张量是什么东西？ ———— 本质上是0维张量，因为张量是容器，哪怕里面只有一个数，它也不会自动变成 Python 数字。

x = torch.tensor([3.0])
print(x, x.shape, x.item())    # tensor([3.]) torch.Size([1]) 3.0
print(torch.tensor(3.0).shape) # torch.Size([])
print()

# 5) 花式索引
#    t[[0, 2]]              → 取第 0 行和第 2 行，shape 是几？
#    t[:, [0, 3]]           → 取第 0 列和第 3 列

print(t[[0, 2]], t[[0, 2]].shape)        # torch.Size([2, 6])
print(t[:, [0, 3]], t[:, [0, 3]].shape)  # torch.Size([4, 2])
print()

# 6) 把 W4 的数据搬过来（今天最重要的一题）
#    读 data/sensor_log.csv 的温度列（复用 W4 的 clean 逻辑，或直接 csv 模块）
#    用 torch.tensor 装进去
#    算 mean / max / min
#    再和 numpy 算一遍对照
#    → 两组数字必须一致（差 < 0.001）

DATA_PATH = os.path.join(BASE_DIR, "data", "sensor_log.csv")
rows = read_csv(DATA_PATH)
rows = rows[1:]
good,right_count,error_count = clean(rows)
t = torch.tensor(good)
arr = np.array(good)

tem_mean = torch.mean(t[:, 1])
tem_max = torch.max(t[:, 1])
tem_min = torch.min(t[:, 1])

hum_mean = torch.mean(t[:, 2])
hum_max = torch.max(t[:, 2])
hum_min = torch.min(t[:, 2])

print(round(np.mean(arr[:, 1]), 2), round(np.max(arr[:, 1]), 2), round(np.min(arr[:, 1]), 2))
print(round(tem_mean.item(), 2), round(tem_max.item(), 2), round(tem_min.item(), 2), round(hum_mean.item(), 2), round(hum_max.item(), 2), round(hum_min.item(), 2))
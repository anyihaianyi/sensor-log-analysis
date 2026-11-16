"""
我先去了解了一下数学理论，我所以为的导数x**2求导得2x，这是标量对标量求导
假设y = x_1**2 + x_2**2，现在是标量对向量求导，得到的是1阶向量
假设y_1 = 3*x_1**2 + 4*x_2**2，y_2 = 2*x_1**2 + x_2**2，现在是向量对向量求导，得到的是2阶矩阵，雅可比矩阵
同理，处理矩阵对矩阵求导，得到3阶张量
那梯度是什么呢 ———— 对多个未知数求导，并将偏导数打包起来，输出是标量 → 梯度（和变量同形状），得到一包斜率
L是损失函数，用于描述最终训练结果的好坏，通过不断微调初始参数来得到不同的损失值，是一个标量
W是一层的权重矩阵，是要训练的参数
什么是链式法则 ———— 反向倒推，沿路相乘，分路相加
"""

import torch

print()
x = torch.arange(4.0)      # 普通数据: [0., 1., 2., 3.]
x.requires_grad_(True)     # ① 举手登记 → x.grad = None（空位留好）
y = 2 * torch.dot(x, x)    # ② 前向运算 → y = 28，菜谱已暗中记录
y.backward()               # ③ 求导动作 → 倒放菜谱，算出 4 个斜率
print(x.grad)              # ④ 领取梯度 → [0., 4., 8., 12.] = 4x
print()

# ── 第 1 题：最小例子，先建立直觉
# x = torch.tensor(3.0, requires_grad=True)
# y = x**2 + 2*x + 1
# y.backward()
# print(x.grad)
#   手算：dy/dx = 2x + 2，在 x=3 时是多少？
#   和打印的对比，必须一致 ———— 一样

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad) # tensor(8.)
print()

# backward 不会推导出 2x+2 这个表达式给你，它只算出 x=3 这一刻的斜率 8（数值）

x = torch.tensor([3.0], requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad) # tensor([8.])
print()

# ── 第 2 题：两个变量
# w = torch.tensor(2.0, requires_grad=True)
# b = torch.tensor(1.0, requires_grad=True)
# loss = (w * 3 + b - 10) ** 2        ← 用 x=3, y=10 这一个点
# loss.backward()
# print(w.grad, b.grad)
#   手推：把 loss 看成 (w*3+b-10)^2，对 w 求导 = 2*(w*3+b-10)*3
#   在 w=2,b=1 时：(6+1-10)=-3，所以 = 2*(-3)*3 = -18
#   对 b 求导 = 2*(-3)*1 = -6
#   和打印的对比 ———— 一样

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
loss = (w * 3 + b - 10) ** 2
loss.backward()
print(w.grad, b.grad) # tensor(-18.) tensor(-6.)
print()

# ── 第 3 题：⭐ 必须撞的坑 ① —— 忘了清零
# w = torch.tensor(1.0, requires_grad=True)
# for i in range(3):
#     loss = w ** 2
#     loss.backward()
#     print(f"第{i}轮 w.grad = {w.grad}")
#   回答：三轮的 grad 是 2, 2, 2 还是 2, 4, 6？为什么？ ———— 是2,4,6 ； PyTorch默认累积梯度
#   然后加一行 w.grad.zero_()，再跑，看变成什么 ———— 2,2,2

w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
     loss = w ** 2
     loss.backward()
     print(f"第{i}轮 w.grad = {w.grad}")
     w.grad.zero_()
print()

w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
     loss = w ** 2
     loss.backward()
     print(f"第{i}轮 w.grad = {w.grad}")
print()

# ── 第 4 题：必须撞的坑 ② —— 把计算图弄断
# x = torch.tensor(2.0, requires_grad=True)
# y = x * 3
# z = y.item() * 2          ← 用 .item() 转成 Python 数字了
# z.backward()              ← 会崩或者 grad 是 None，看它报什么
#   回答：为什么用了 .item() 就断了？ ———— item会把原来张量里的requires_grad、grad_fn等等丢掉
#   再试一次，把 y 直接 .numpy() 会怎样？ ———— 也不行，需要先.detach()来截断张量的梯度运算，才能转为numpy

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.item() * 2 # 但是y本身仍然是张量
# z.backward() # AttributeError: 'float' object has no attribute 'backward'
# z = y.numpy() * 2
# z.backward() # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

# ── 第 5 题：no_grad
# with torch.no_grad():
#     造一个 requires_grad=True 的张量，做运算，打印结果的 requires_grad
#   回答：什么时候需要 no_grad？（提示：W7 的推理阶段全部用它） ———— 记录运算过程是需要内存的，训练过程很吃显存，且有些场景不需要.backward

# no_grad() 是又一种"关掉梯度"的开关，但作用范围不同：detach 是剪断一根菜谱线，no_grad 是让整间厨房在这段时间都不写菜谱。
x = torch.tensor(3.0, requires_grad=True)
with torch.no_grad():
    # 块内：张量运算照常执行、数值分毫不差
    # 但一律不记录计算图——产物没有 grad_fn，不能 backward
    y = x**2 + 2*x + 1
print(y)
# y.backward()

# 块外：自动恢复记录


"""
开关                        作用               范围效果                        比喻
x.detach()                 单个张量（点）     这个张量当常数，其余照常记录      剪断一根菜谱线
with torch.no_grad():      整段代码（面）     块内所有运算都不记录             厨房暂停写菜谱
x.item() / .numpy()        搬数值出张量世界    彻底退群                       带着数字退群
x.requires_grad_(False)    改张量自身的标志    从此不再举手                    注销旋钮身份
"""
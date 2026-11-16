import torch
import time
import os
from torchvision.io import read_image



print()
# ── 造数据：y = 3x + 2 + 噪声，200 个点
# torch.manual_seed(42) ← 在计算机中，所谓的“随机数”其实是“伪随机数”，它们是通过一个复杂的数学公式计算出来的。这个公式的初始输入值就被称为“种子（seed）”
# x = torch.randn(200, 1)
# y = 3 * x + 2 + torch.randn(200, 1) * 0.1
# 打印 x.shape、y.shape
torch.manual_seed(42)
x = torch.randn(200, 1)
y = 3 * x + 2 + torch.randn(200, 1) * 0.1
print(x.shape, y.shape)

# ── 参数：两个要学的数
# w = torch.randn(1, requires_grad=True) <- 生成一个包含1个随机数的张量，且这个随机数服从标准正态分布（均值为 0，方差为 1）
# b = torch.zeros(1, requires_grad=True)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ── 优化器（直接用现成的，不自己写）
# optimizer = torch.optim.SGD([w, b], lr=0.1) <- 定义一个“优化器”（Optimizer），用来告诉神经网络“如何根据计算出的误差，去调整参数（w 和 b），从而让模型变得更准确”
optimizer = torch.optim.SGD([w, b], lr=0.1)
# 如果说模型是“大脑”，损失函数（Loss）是“考官打分”，那么优化器就是“纠错老师”，它负责根据考官的反馈，具体去修改大脑里的记忆（参数）
"""
SGD：全称是 Stochastic Gradient Descent（随机梯度下降）。这是最经典、最基础的优化算法。它通过计算误差对参数的“梯度”（即变化趋势），然后让参数朝着让误差变小的方向移动。
在简单的机器学习模型（比如线性回归y = wx + b）中，w 代表权重（Weight），b 代表偏置（Bias），[w, b]是告诉优化器需要被优化的参数列表
lr 全称是 Learning Rate（学习率），决定了每次调整参数时，步子迈得有多大。
"""

# ── 训练循环：五步，顺序不能错
# for epoch in range(100):
#     ① 预测：pred = w * x + b
#     ② 算 loss：loss = ((pred - y) ** 2).mean()
#     ③ 清零：optimizer.zero_grad()
#     ④ 反向：loss.backward()
#     ⑤ 更新：optimizer.step()
#     每 10 轮打印一次 epoch 和 loss
for epoch in range(100):
    pred = w * x + b
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%10 == 0:
        print(epoch, loss.item())
"""
优化器（如 SGD）的底层数学基础是微积分。它不是盲目地试错，而是通过计算“坡度”（梯度），直接指明下山的方向。
    在数学上，梯度就是偏导数。它表示的是：在当前这个点，如果参数 w 增加一点点，Loss 会增加还是减少？变化的速率是多少？
    如果梯度是正的，说明 w 变大，Loss 也会变大。
    如果梯度是负的，说明 w 变大，Loss 会变小。
优化器执行的核心公式极其简单：新参数=老参数−学习率(lr)×梯度
    减号（-）：因为梯度指向的是 Loss 增大最快的方向，我们要加个负号，让它朝着 Loss 减小的方向走。
"""

# ── 检查结果
# 打印最终的 w 和 b
#   → 期望：w ≈ 3，b ≈ 2（差多少算正常？自己判断）
print(w, b)
print()

# ── ⭐ 搬到 GPU 上再跑一遍
# x = x.to("cuda"); y = y.to("cuda"); w = ...; b = ...
# 全部搬过去，重跑一遍
# 在循环里打印一次 pred.device，确认是 cuda:0
# 回答：GPU 版比 CPU 版快吗？（200 个点太小了，可能反而更慢——这是正常的，说清为什么） ———— 200 个点的数据，GPU 极可能更慢——因为把数据在内存和显存之间搬来搬去的开销，远大于这点计算量。这不是猜，是常识：GPU 赢在并行，200 个数根本没有并行的余地。
x = x.to("cuda")
y = y.to("cuda")
device = "cuda"
w = torch.randn(1, requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)
optimizer = torch.optim.SGD([w, b], lr=0.1)

for epoch in range(100):
    pred = w * x + b                # pred就是向前传播
    loss = ((pred - y) ** 2).mean() # pred-y用于计算误差，平方用于扩大误差并消除负号，求平均误差
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%10 == 0:
        print(epoch, loss.item())
    if epoch == 99:
        print(pred.device)
print(w, b)
print()

"""
应对“梯度相同/对称”（鞍点问题）：
    基础 SGD 的做法：直接卡死，原地不动。
    高级优化器的做法：引入随机性。每次下山不看整座山，只随机抓一把土（Mini-batch 小批量梯度下降）。这把土的坡度是不均匀的，这种“噪声”能打破对称，推着你偏离中心，从而继续下山。
应对“选错方向/掉进浅坑”（局部最优问题）：
    基础 SGD 的做法：掉进浅坑后，发现四周都是上坡（梯度指向坑外），它以为到底了，就死在这个浅坑里了。
    高级优化器的做法（引入“动量 Momentum”）：它给下山的人加上了 “物理惯性”。如果你一直往右下冲，你的速度会越来越快，惯性极大。当你掉进右下的浅坑时，巨大的惯性会直接让你“冲”出这个浅坑，越过旁边的山坡，最终滚进左下那个真正的全局深谷里！
应对“有的方向步子该大，有的方向步子该小”：
    高级优化器的做法（自适应学习率，如 Adam, RMSprop）：优化器会记住每个参数过去的历史梯度。对于一直梯度很大的参数，自动把它的 lr 调小；对于一直梯度很小的参数，自动把它的 lr 调大。让每个人都能以最合适的步幅下山。
"""

def train(device, epochs=100):
    w = torch.randn(1, requires_grad=True, device=device)
    b = torch.zeros(1, requires_grad=True, device=device)
    opt = torch.optim.SGD([w, b], lr=0.1)
    x_, y_ = x.to(device), y.to(device)

    t0 = time.time()
    for _ in range(epochs):
        pred = w * x_ + b
        loss = ((pred - y_) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    cost = time.time() - t0

    print(f'{device}: {cost*1000:.2f} ms, w={w.item():.4f}, b={b.item():.4f}')
    return cost

print()
train("cpu")
train("cuda")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_PATH  = os.path.join(BASE_DIR, "figs", "temp_humidity.png")

img = read_image(FIG_PATH)
print(img.shape, img.dtype)
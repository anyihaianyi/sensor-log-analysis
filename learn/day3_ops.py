import torch

a = torch.tensor([[1., 2.],
                  [3., 4.]])
b = torch.tensor([[5., 6.],
                  [7., 8.]])

# 1) 三种"乘"，结果完全不同
#    a + b、a - b         → 逐元素
#    a * b                → 逐元素乘（不是矩阵乘！）
#    a @ b                → 矩阵乘（也可以写 torch.matmul(a, b)）
#    手算 a @ b 的结果，再和打印的对比

print(a + b)
print(a - b)
print(a * b)
print(a @ b)

# 2) 形状不对会怎样
#    造 (2,3) 和 (2,3)，做 @ —— 应该崩。把报错原文抄进注释
#    改成 (2,3) @ (3,2) —— 能算，结果 shape 是几？

a = torch.tensor([[1, 2, 3],
                  [2, 3, 4]])
b = torch.tensor([[2, 3, 4],
                  [3, 4, 5]])
# print(a @ b)  # RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)

c = torch.tensor([[1, 2, 3], [4, 5, 6]])          # (2,3)
d = torch.tensor([[1, 2], [3, 4], [5, 6]])        # (3,2)
print((c @ d).shape)


# 3) 广播（今天最难的一块）
#    x = torch.arange(3).reshape(3, 1)      → shape (3,1)
#    y = torch.arange(4).reshape(1, 4)      → shape (1,4)
#    z = x + y                              → 打印 z 和 z.shape
#    回答：为什么两个"都不完整"的形状能加出 (3,4)？ ———— “最后一维 1 可以拉成 4，倒数第二维 3 可以拉成 3”

x = torch.arange(3).reshape(3, 1)
y = torch.arange(4).reshape(1, 4)
print(x.shape)
print(y.shape)
z = x + y
print(z)
print(z.shape) 

# 4) 广播失败的例子（必须撞一次）
#    造 (3,2) 和 (3,3) 相加 —— 崩。抄报错
#    回答：为什么 (3,1)+(1,4) 行，而 (3,2)+(3,3) 不行？规则是什么？ 

"""
① 从最后一维开始，往左逐位比较
② 每一对维度，要么相等，要么其中一个是 1（是 1 的那个会被拉伸）
③ 任何一个维度都不满足 → 崩
"""

x = torch.arange(6).reshape(3, 2)
y = torch.arange(9).reshape(3, 3)
# print(x + y) # RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

# 5) reshape vs view
#    造 t = torch.arange(12)
#    分别 t.reshape(3,4) 和 t.view(3,4)，都能用
#    然后 t2 = t.reshape(3,4).T        ← 转置
#    t2.view(12)                        ← 应该崩，抄报错
#    t2.reshape(12)                     ← 能跑
#    回答：view 和 reshape 差在哪？ ———— 两个都可以用来改变形状，view 保证零拷贝（做不到就报错）；reshape 保证能成功（做不到就偷偷拷贝）

t = torch.arange(12)
print(t.reshape(3,4))
print(t.view(3,4))
t2 = t.reshape(3, 4).T
# print(t2.view(12)) # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
print(t2.reshape(12))

# 6) 就地修改：下划线后缀
#    m = torch.ones(2, 2)
#    n = m.add(1)        → 打印 m 和 n
#    p = m.add_(1)       → 打印 m 和 p
#    回答：下划线是什么意思？为什么 PyTorch 要专门提供这种写法？ ———— 下划线是原地操作的意思，可以节省内存

m = torch.ones(2, 2)
n = m.add(1)
print(m)
print(n)
p = m.add_(1)
print(m)
print(p) # 不改变m的内存，直接改变内存存的值，m和p共用一个内存

# 7) 统计函数：dim 就是 numpy 的 axis
#    t = torch.arange(12).reshape(3, 4).float()
#    打印 t.mean()、t.sum()
#    打印 t.mean(dim=0) 和它的 shape    ← 对应 numpy 的 axis=0
#    打印 t.mean(dim=1) 和它的 shape
#    回答：dim=0 消失的是哪一维？（和 W4 D6 你验证过的一模一样） ———— 消失的是行，就是对应每一列的均值

t = torch.arange(12).reshape(3, 4).float()
print(t.mean())
print(t.sum())
print(t.mean(dim=0), t.mean(dim=0).shape)
print(t.mean(dim=1), t.mean(dim=1).shape)

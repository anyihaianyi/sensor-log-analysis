import torch

# ── day5第 1 题
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad)


# ── day5第 3 题
w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    loss = w ** 2
    loss.backward()
    print(f"第{i}轮 w.grad = {w.grad}")

w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    loss = w ** 2
    loss.backward()
    print(f"第{i}轮 w.grad = {w.grad}")
    w.grad.zero_()
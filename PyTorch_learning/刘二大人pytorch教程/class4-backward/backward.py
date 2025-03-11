import torch
import matplotlib.pyplot as plt

#3、画出二次模型y = w1x²+w2x + b，损失函数loss = (ŷ - y)²的计算图，并且手动推导反向传播的过程
#最后用pytorch的代码实现。

# y = w1x^2 + w2x + b
# 正确值：w1 = 2, w2 = 3, b = 4

x_data = [0.0, 1.0, 2.0, 3.0]
y_data = [4.0, 9.0, 18.0, 31.0]

w1_data = []
w2_data = []
b_data = []

epoch_data = []
epoch = 1

# 创建Tensor节点
w1 = torch.Tensor([10.0])
w2 = torch.Tensor([10.0])
b = torch.Tensor([10.0])
# 设置计算梯度
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

for epoch in range(1000):

    # 每次都能拿到一个梯度，直接全部都用
    for x, y in zip(x_data, y_data):
        # 构建图
        l = ((w1 * (x ** 2) + w2 * x + b) - y) ** 2  # 其实就是利用反向传播来求这个式子在各个权重方向的偏导
        # 反馈，更新grad（梯度）值
        l.backward()
        # 根据梯度下降法公式更新权重
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        print("epoch:", epoch)
        print("W1:", w1.data)
        print("W2:", w2.data)
        print("b:", b.data)
        w1_data.append(w1.data.item())  # 这里也可以w_data.append(w.data),plt也能识别出来Tensor里面的数值
        w2_data.append(w2.data.item())
        b_data.append(b.data.item())
        epoch_data.append(epoch)
        epoch += 1
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

plt.plot(epoch_data, w1_data, "g", label="W1")
plt.plot(epoch_data, w2_data, "r", label="W2")
plt.plot(epoch_data, b_data, label="b")

plt.xlabel("Epoch")
plt.show()
#---------------------------------------------------------------------------------
# y = wx模型
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_data = []
epoch_data = []
epoch = 1

# 创建Tensor节点
w = torch.Tensor([66.0])
# 设置计算梯度
w.requires_grad = True

for epoch in range(1000):

    # 每次都能拿到一个梯度，直接全部都用
    for x, y in zip(x_data, y_data):
        # 构建图
        l = (w * x - y) ** 2
        # 反馈，更新grad（梯度）值
        l.backward()
        # 根据梯度下降法公式更新权重
        w.data = w.data - 0.01 * w.grad.data
        print("epoch:", epoch)
        print("梯度w.grad.item", w.grad.item())
        print("W:", w.data)
        w_data.append(w.data.item())  # 这里也可以w_data.append(w.data),plt也能识别出来Tensor里面的数值
        epoch_data.append(epoch)
        epoch += 1
        w.grad.data.zero_()

print(epoch_data)
print(w_data)

plt.plot(epoch_data, w_data)
plt.xlabel("Epoch")
plt.ylabel("W")
plt.show()
#问题1：手动推导线性模型 y=w*x，损失函数 loss=(ŷ-y)²，当数据点 x=2, y=4 时的反向传播过程
w = torch.tensor([1.0], requires_grad=True)
x, y = 2.0, 4.0

loss = (w * x - y) ** 2
loss.backward()
print("梯度:", w.grad.item())  # 输出 -8.0

#问题2：手动推导线性模型 y=w*x + b，损失函数 loss=(ŷ-y)²，当数据点 x=1, y=2 时的反向传播过程
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
x, y = 1.0, 2.0

loss = (w * x + b - y) ** 2
loss.backward()
print("w的梯度:", w.grad.item())  # 输出 -2.0
print("b的梯度:", b.grad.item())  # 输出 -2.0
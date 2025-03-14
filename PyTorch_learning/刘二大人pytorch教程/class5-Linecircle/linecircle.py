import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# 1. 准备数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])  # 输入特征 (3样本,1特征)
y_data = torch.Tensor([[2.0], [4.0], [6.0]])  # 标签数据 (3样本,1输出)


# 2. 定义模型（每次训练前重置参数）
def create_model():
    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)  # 输入1维，输出1维

        def forward(self, x):
            return self.linear(x)

    return LinearModel()


# 3. 定义训练函数（输入优化器名称，返回损失列表）
def train_with_optimizer(optimizer_name, lr=0.01, epochs=100):
    model = create_model()  # 每次训练前创建新模型（参数重置）
    criterion = nn.MSELoss()  # 均方误差损失

    # 根据优化器名称选择优化器
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_name == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Rprop':
        optimizer = optim.Rprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    losses = []  # 记录每个epoch的损失值
    for epoch in range(epochs):
        y_pred = model(x_data)  # 前向传播
        loss = criterion(y_pred, y_data)
        losses.append(loss.item())

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    return losses


# 4. 训练所有优化器并记录损失
optimizers = ['SGD', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'Rprop']
epochs = 1234
losses_dict = {}  # 保存每个优化器的损失曲线

for opt_name in optimizers:
    losses = train_with_optimizer(opt_name, lr=0.01, epochs=epochs)
    losses_dict[opt_name] = losses
    print(f"{opt_name} 训练完成，最终损失: {losses[-1]:.4f}")
再用Adagrad Adam adamax ASGD RMSprop Rprop SGD七种优化器实现，并分别输出loss关于time的曲线图，给出完整代码（带通俗易懂注释）
# 5. 绘制损失曲线
plt.figure(figsize=(10, 6))
for opt_name in optimizers:
    plt.plot(range(epochs), losses_dict[opt_name], label=opt_name)

plt.xlabel('Epoch')  # 横轴：训练轮次
plt.ylabel('Loss')  # 纵轴：损失值
plt.title('不同优化器的损失曲线对比')  # 标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()
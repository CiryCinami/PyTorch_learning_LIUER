import matplotlib.pyplot as plt
import random

"""-----------------------------------------
   批量梯度下降 (Batch Gradient Descent)
-----------------------------------------"""


# 配置1：批量梯度下降参数
def run_batch_gd():
    # 数据集
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    w = 1.0  # 初始权重
    lr = 0.01

    # 模型定义
    def forward(x):
        return x * w

    # 全体样本平均损失
    def loss():
        total = 0
        for x, y in zip(x_data, y_data):
            total += (forward(x) - y) ** 2
        return total / len(x_data)

    # 全体样本平均梯度
    def gradient():
        g = 0
        for x, y in zip(x_data, y_data):
            g += 2 * x * (x * w - y)
        return g / len(x_data)  # 关键差异点：平均梯度

    # 训练循环
    epochs, loss_batch = [], []
    print("【Batch GD】初始预测 x=4:", forward(4))
    for epoch in range(100):
        w -= lr * gradient()  # 关键差异点：每个epoch更新一次
        epochs.append(epoch)
        loss_batch.append(loss())

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.plot(epochs, loss_batch, 'b')
    plt.title("Batch GD Loss")
    print("【Batch GD】最终预测 x=4:", forward(4))


"""-----------------------------------------
   随机梯度下降 (Stochastic Gradient Descent)
-----------------------------------------"""


# 配置2：随机梯度下降参数
def run_sgd():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    w = 1.0
    lr = 0.01

    def forward(x):
        return x * w

    # 单样本梯度
    def single_gradient(x, y):
        return 2 * x * (x * w - y)  # 关键差异点：单个样本梯度

    # 训练循环
    epochs, loss_sgd = [], []
    print("\n【SGD】初始预测 x=4:", forward(4))
    for epoch in range(100):
        epoch_loss = 0
        for x, y in zip(x_data, y_data):
            grad = single_gradient(x, y)
            w -= lr * grad  # 关键差异点：逐样本更新
            epoch_loss += (forward(x) - y) ** 2
        epochs.append(epoch)
        loss_sgd.append(epoch_loss / len(x_data))

    plt.subplot(132)
    plt.plot(epochs, loss_sgd, 'r')
    plt.title("SGD Loss")
    print("【SGD】最终预测 x=4:", forward(4))


"""-----------------------------------------
   小批量梯度下降 (Mini-batch GD)
-----------------------------------------"""


# 配置3：小批量梯度下降参数
def run_mini_batch():
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_data = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    w = 1.0
    lr = 0.01
    batch_size = 2

    def forward(x):
        return x * w

    # 小批量梯度计算
    def batch_gradient(batch):
        g = 0
        for x, y in batch:
            g += 2 * x * (x * w - y)
        return g / len(batch)  # 关键差异点：小批量平均

    # 训练循环
    epochs, loss_mini = [], []
    print("\n【Mini-batch】初始预测 x=4:", forward(4))
    for epoch in range(100):
        # 数据洗牌
        shuffled = list(zip(x_data, y_data))
        random.shuffle(shuffled)

        # 分批次更新
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            grad = batch_gradient(batch)
            w -= lr * grad  # 关键差异点：批次更新

        # 记录整体损失
        total_loss = sum((forward(x) - y) ** 2 for x, y in shuffled) / len(shuffled)
        epochs.append(epoch)
        loss_mini.append(total_loss)

    plt.subplot(133)
    plt.plot(epochs, loss_mini, 'g')
    plt.title("Mini-batch Loss")
    print("【Mini-batch】最终预测 x=4:", forward(4))


"""-----------------------------------------
   执行与可视化
-----------------------------------------"""
if __name__ == "__main__":
    plt.figure(figsize=(15, 5))
    run_batch_gd()
    run_sgd()
    run_mini_batch()
    plt.tight_layout()
    plt.show()
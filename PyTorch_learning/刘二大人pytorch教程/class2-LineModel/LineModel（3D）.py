import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 训练数据
x_data = [1.135415, 2.463546, 3.5498646]
y_data = [2.2, 4.8, 6.4246]

# 生成参数网格
w_cor = np.arange(0.0, 4.0, 0.1)
b_cor = np.arange(-2.0, 2.1, 0.1)
w, b = np.meshgrid(w_cor, b_cor)

# 计算MSE损失
mse = np.zeros(w.shape)
for x, y in zip(x_data, y_data):
    y_pred = w * x + b  # 模型预测
    mse += (y_pred - y)**2  # 累加平方误差
mse /= len(x_data)  # 计算均值

# 创建3D绘图
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  # 更现代的3D坐标轴创建方式

# 曲面图参数配置
surf = ax.plot_surface(
    w, b, mse,
    rstride=1,  # 行步长（分辨率）
    cstride=1,  # 列步长
    cmap=cm.rainbow,
    antialiased=True
)

# 坐标轴标签
ax.set_xlabel('Weight (w)', fontsize=12, labelpad=10)
ax.set_ylabel('Bias (b)', fontsize=12, labelpad=10)
ax.set_zlabel('Loss', fontsize=12, labelpad=10)

# 颜色条
fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.12)

# 视角调整
ax.view_init(elev=30, azim=240)  # 仰角30度，方位角240度

plt.show()
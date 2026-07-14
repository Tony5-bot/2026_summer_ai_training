# ============================================================================
# 线性回归的从零开始实现（对应《动手学深度学习》第3.2节）
# 在线地址：https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html
# ============================================================================

import random
import torch
from d2l import torch as d2l


# ============================================================================
# 1. 生成数据集
# ============================================================================
# 构造一个包含1000个样本的人工数据集，特征维度为2
# 真实权重 w = [2, -3.4]，真实偏置 b = 4.2
# 添加均值为0、标准差为0.01的高斯噪声

def synthetic_data(w, b, num_examples):  #@save
    """生成 y = Xw + b + 噪声的合成数据"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 从标准正态分布采样特征
    y = torch.matmul(X, w) + b                       # 计算线性输出
    y += torch.normal(0, 0.01, y.shape)              # 添加噪声项
    return X, y.reshape((-1, 1))                     # 将y转为列向量

# 设置真实参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])

# 可视化第二个特征与标签的关系（可选）
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(),
#                 labels.detach().numpy(), 1)


# ============================================================================
# 2. 读取数据集
# ============================================================================
# 实现一个函数，每次从数据集中随机抽取一个小批量样本

def data_iter(batch_size, features, labels):
    """生成大小为batch_size的小批量数据迭代器"""
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机打乱样本索引（对所有样本的读取顺序是随机的）
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 设置批量大小
batch_size = 10

# 读取第一个小批量数据样本并打印形状
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# ============================================================================
# 3. 初始化模型参数
# ============================================================================
# 权重从均值为0、标准差为0.01的正态分布中采样，偏置初始化为0
# 设置 requires_grad=True 以自动计算梯度

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# ============================================================================
# 4. 定义模型
# ============================================================================
# 线性回归模型：y = Xw + b

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# ============================================================================
# 5. 定义损失函数
# ============================================================================
# 均方误差损失（平方损失的一半，方便求导）

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# ============================================================================
# 6. 定义优化算法
# ============================================================================
# 小批量随机梯度下降（Mini-batch SGD），使用参数的梯度更新参数
# 每个参数减去 学习率 × 梯度

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():  # 更新参数时不需要计算梯度
        for param in params:
            param -= lr * param.grad / batch_size  # 沿负梯度方向更新
            param.grad.zero_()                      # 梯度清零，防止累积


# ============================================================================
# 7. 训练
# ============================================================================
# 完整训练流程：正向传播 → 计算损失 → 反向传播 → 更新参数
# 训练多个epoch，每个epoch遍历整个数据集一次

lr = 0.03                           # 学习率
num_epochs = 3                      # 迭代周期数
net = linreg                        # 模型
loss = squared_loss                 # 损失函数

for epoch in range(num_epochs):
    # 遍历每个小批量
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)   # 正向传播：计算小批量损失
        # 因为l的形状是(batch_size, 1)，需要求和得到标量再进行反向传播
        l.sum().backward()           # 反向传播：计算梯度
        sgd([w, b], lr, batch_size) # 使用梯度更新参数

    # 每个epoch结束后，计算整个数据集上的损失
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 打印估计参数与真实参数的差距
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

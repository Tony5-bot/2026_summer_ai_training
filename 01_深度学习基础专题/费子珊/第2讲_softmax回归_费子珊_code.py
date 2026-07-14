# ============================================================================
# Softmax回归的从零开始实现（对应《动手学深度学习》第3.5节 + 第3.6节）
# 在线地址：
#   - 3.5: https://zh.d2l.ai/chapter_linear-networks/image-classification-dataset.html
#   - 3.6: https://zh.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html
# ============================================================================

import torch
from torch import nn
from d2l import torch as d2l


# ============================================================================
# 第一部分：图像分类数据集 — Fashion-MNIST（对应书上 3.5 节）
# ============================================================================

# ---------------------------------------------------------------------------
# 1. 读取数据集
# ---------------------------------------------------------------------------
# Fashion-MNIST 包含 10 个类别的 28×28 灰度图像
# 训练集 60000 张，测试集 10000 张

batch_size = 256

# 使用 d2l 封装的函数加载 Fashion-MNIST 数据集
# ToTensor 将 PIL 图像转为 32 位浮点数，并将像素值从 0~255 缩放到 0~1
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# ============================================================================
# 第二部分：Softmax回归从零开始实现（对应书上 3.6 节）
# ============================================================================

# ---------------------------------------------------------------------------
# 1. 初始化模型参数
# ---------------------------------------------------------------------------
# Fashion-MNIST 每张图像是 28×28 = 784 像素（展平后）
# 有 10 个类别，因此权重矩阵形状为 (784, 10)，偏置形状为 (10,)

num_inputs = 784     # 输入特征数（28×28像素展平）
num_outputs = 10     # 输出类别数

# 权重从均值为0、标准差为0.01的正态分布中随机初始化
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 偏置初始化为0
b = torch.zeros(num_outputs, requires_grad=True)


# ---------------------------------------------------------------------------
# 2. 定义 Softmax 操作
# ---------------------------------------------------------------------------
# Softmax 将未规范化的预测（logits）转换为概率分布
# 公式：softmax(X)_{ij} = exp(X_{ij}) / sum_k(exp(X_{ik}))

def softmax(X):
    """对矩阵X的每一行进行softmax运算"""
    X_exp = torch.exp(X)                        # 对每个元素求指数
    partition = X_exp.sum(1, keepdim=True)       # 对每行求和（保持维度用于广播）
    return X_exp / partition                     # 广播除法：每行除以对应的和


# ---------------------------------------------------------------------------
# 3. 定义模型
# ---------------------------------------------------------------------------
# Softmax 回归模型：将输入展平 → 线性变换 → softmax
# 输入 X 形状为 (batch_size, 28, 28)，先展平为 (batch_size, 784)

def net(X):
    """Softmax回归模型"""
    # 使用reshape将每张图像展平为长度为784的向量
    # -1 表示让PyTorch自动推断该维度的大小（即batch_size）
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# ---------------------------------------------------------------------------
# 4. 定义交叉熵损失函数
# ---------------------------------------------------------------------------
# 交叉熵损失：L = -log(y_hat[y])，即正确类别的负对数概率
# 实现时注意：这里直接用 softmax 输出算 log，当概率接近 0 时可能有
# log(0) 的数值问题。在实际工程中应使用 nn.CrossEntropyLoss

def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    # y_hat[range(len(y_hat)), y] 获取每个样本在其正确类别上的预测概率
    # -log 得到该样本的交叉熵损失
    return -torch.log(y_hat[range(len(y_hat)), y])


# ---------------------------------------------------------------------------
# 5. 分类精度
# ---------------------------------------------------------------------------
# 精度 = 正确预测的数量 / 总预测数量
# 这是分类任务最常用的评估指标

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 如果y_hat是矩阵（概率分布），取每行最大值的索引作为预测类别
        y_hat = y_hat.argmax(axis=1)
    # 将预测类别与真实标签比较，求和得到正确预测的数量
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  #@save
    """计算模型在指定数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式（不计算dropout等）
    metric = d2l.Accumulator(2)  # 累加 [正确预测数, 总预测数]（使用d2l内置的Accumulator）
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]  # 精度 = 正确数 / 总数


# ---------------------------------------------------------------------------
# 6. 训练
# ---------------------------------------------------------------------------
# 训练一个epoch的函数

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 如果模型是nn.Module，设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 累加器：训练损失总和, 训练准确数, 样本数
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)                        # 前向传播
        l = loss(y_hat, y)                    # 计算损失
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置优化器
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用自定义优化器
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章），使用 d2l.Animator 实时绘制训练曲线"""
    # 使用 d2l 内置的 Animator（基于 matplotlib，比自定义版本更稳定）
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 在测试集上评估精度
        test_acc = evaluate_accuracy(net, test_iter)
        # 添加数据点到动画曲线
        animator.add(epoch + 1, train_metrics + (test_acc,))
        # 打印训练信息
        train_loss, train_acc = train_metrics
        print(f'epoch {epoch + 1}: train_loss={train_loss:.3f}, '
              f'train_acc={train_acc:.3f}, test_acc={test_acc:.3f}')
    print(f'训练完成！最终训练损失={train_metrics[0]:.3f}, '
          f'训练精度={train_metrics[1]:.3f}, 测试精度={test_acc:.3f}')
    # 确保至少有一个预测是正确的
    assert test_acc >= 0.7, f'测试精度={test_acc:.3f} 太低，请检查实现！'


# ---------------------------------------------------------------------------
# 7. 小批量随机梯度下降（自定义优化器）
# ---------------------------------------------------------------------------

lr = 0.1   # 学习率

def updater(batch_size):
    """小批量随机梯度下降更新参数"""
    return d2l.sgd([W, b], lr, batch_size)


# ---------------------------------------------------------------------------
# 8. 开始训练
# ---------------------------------------------------------------------------
# 训练 10 个 epoch

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# ---------------------------------------------------------------------------
# 9. 预测
# ---------------------------------------------------------------------------

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    # 从测试集中取一批数据
    for X, y in test_iter:
        break
    # 获取真实标签名称
    trues = d2l.get_fashion_mnist_labels(y)
    # 获取模型预测标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    # 组合标题：真实标签 + 预测标签
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # 展示前n个样本
    d2l.show_images(
        X[0:n].reshape((-1, 28, 28)), 1, n, titles=titles[0:n])
    # 在脚本模式下需要显式显示图像（Jupyter中会自动显示）
    d2l.plt.show()

# 对新样本进行预测并可视化结果
predict_ch3(net, test_iter)

"""
第四章 多层感知机 — 重要代码合集
==============================================
基于《动手学深度学习》(d2l.ai) 第4章（4.1–4.10）
整理人：费子珊

══════════════════════ 使用方法 ═══════════════════════

  本文件将每小节的代码封装为独立函数，互不干扰。
  想运行哪一节，就把对应那行的注释去掉即可。

  例如：只想运行 4.6 Dropout
    → 去掉 section_4_6_dropout() 前面的 #

  所有函数定义在上面，调用入口在最底部。

══════════════════════════════════════════════════════

包含内容：
  4.2  多层感知机的从零开始实现
  4.5  权重衰减（从零实现 + 简洁实现）
  4.6  暂退法 Dropout（从零实现 + 简洁实现）
  4.10 实战Kaggle比赛：预测房价

注：4.3节（简洁实现MLP）的代码未纳入。
"""

import torch
from torch import nn
from d2l import torch as d2l


# ============================================================
# 4.2  多层感知机的从零开始实现
# ============================================================

def section_4_2_mlp_scratch():
    """从零实现单隐藏层MLP，在Fashion-MNIST上训练"""
    print("=" * 50)
    print("4.2  多层感知机的从零开始实现")
    print("=" * 50)

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # ---------- 初始化模型参数 ----------
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    # ---------- 激活函数 ----------
    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)

    # ---------- 模型 ----------
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)  # "@"代表矩阵乘法
        return (H @ W2 + b2)

    # ---------- 损失函数 ----------
    loss_fn = nn.CrossEntropyLoss()  # 默认 reduction='mean'，返回批次平均损失

    # ---------- 训练 ----------
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)

    for epoch in range(num_epochs):
        # 训练阶段
        train_loss_sum, train_acc_sum, n_batches = 0.0, 0.0, 0
        for X, y in train_iter:
            updater.zero_grad()
            y_hat = net(X)
            l = loss_fn(y_hat, y)       # 批次平均损失
            l.backward()
            updater.step()
            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().mean().item()
            n_batches += 1
        train_loss = train_loss_sum / n_batches
        train_acc = train_acc_sum / n_batches

        # 测试阶段
        test_acc_sum, m = 0.0, 0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                test_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
                m += y.shape[0]
        test_acc = test_acc_sum / m

        print(f'epoch {epoch + 1:2d}, '
              f'loss {train_loss:.4f}, '
              f'train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')

    # ---------- 预测展示 ----------
    print("\n>>> 测试集样本预测展示")
    test_iter_demo, _ = d2l.load_data_fashion_mnist(6)
    X, y = next(iter(test_iter_demo))
    with torch.no_grad():
        preds = net(X).argmax(dim=1)

    # Fashion-MNIST 类别名称
    labels = ['T恤', '裤子', '毛衣', '连衣裙', '外套',
              '凉鞋', '衬衫', '运动鞋', '包', '靴子']
    print('-' * 50)
    for i in range(len(X)):
        print(f'  样本{i+1}:  预测=[{labels[preds[i].item()]}]  真实=[{labels[y[i].item()]}]  '
              f'{"正确" if preds[i] == y[i] else "错误"}')
    print('-' * 50)


# ============================================================
# 4.5  权重衰减（Weight Decay）
# ============================================================

def section_4_5_weight_decay():
    """权重衰减：从零实现 + 简洁实现，在高维线性回归上演示"""
    print("=" * 50)
    print("4.5  权重衰减")
    print("=" * 50)

    # ---- 生成高维线性回归数据 ----
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    # ---------- 初始化模型参数 ----------
    def init_params():
        w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        return [w, b]

    # ---------- L2范数惩罚 ----------
    def l2_penalty(w):
        return torch.sum(w.pow(2)) / 2

    # ---------- 训练（手动实现权重衰减） ----------
    def train(lambd):
        w, b = init_params()
        net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
        num_epochs, lr = 100, 0.003
        animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                                xlim=[5, num_epochs], legend=['train', 'test'])
        for epoch in range(num_epochs):
            for X, y in train_iter:
                # 增加了L2范数惩罚项
                l = loss(net(X), y) + lambd * l2_penalty(w)
                l.sum().backward()
                d2l.sgd([w, b], lr, batch_size)
            if (epoch + 1) % 5 == 0:
                animator.add(epoch + 1,
                             (d2l.evaluate_loss(net, train_iter, loss),
                              d2l.evaluate_loss(net, test_iter, loss)))
        print('w的L2范数是：', torch.norm(w).item())

    print("\n>>> 不使用权重衰减（λ=0）")
    train(lambd=0)

    print("\n>>> 使用权重衰减（λ=3）")
    train(lambd=3)

    # ---------- 简洁实现 ----------
    def train_concise(wd):
        net = nn.Sequential(nn.Linear(num_inputs, 1))
        for param in net.parameters():
            param.data.normal_()
        loss = nn.MSELoss(reduction='none')
        num_epochs, lr = 100, 0.003
        # 偏置参数没有衰减
        trainer = torch.optim.SGD([
            {"params": net[0].weight, 'weight_decay': wd},
            {"params": net[0].bias}], lr=lr)
        animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                                xlim=[5, num_epochs], legend=['train', 'test'])
        for epoch in range(num_epochs):
            for X, y in train_iter:
                trainer.zero_grad()
                l = loss(net(X), y)
                l.mean().backward()
                trainer.step()
            if (epoch + 1) % 5 == 0:
                animator.add(epoch + 1,
                             (d2l.evaluate_loss(net, train_iter, loss),
                              d2l.evaluate_loss(net, test_iter, loss)))
        print('w的L2范数：', net[0].weight.norm().item())

    print("\n>>> 简洁实现（wd=0）")
    train_concise(0)

    print("\n>>> 简洁实现（wd=3）")
    train_concise(3)


# ============================================================
# 4.6  暂退法 Dropout
# ============================================================

def section_4_6_dropout():
    """Dropout：从零实现 + 简洁实现，在Fashion-MNIST上训练"""
    print("=" * 50)
    print("4.6  暂退法 Dropout")
    print("=" * 50)

    # ---------- 从零实现 dropout_layer ----------
    def dropout_layer(X, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1:
            return torch.zeros_like(X)
        if dropout == 0:
            return X
        mask = (torch.rand(X.shape) > dropout).float()
        return mask * X / (1.0 - dropout)

    # ---------- 测试 dropout_layer ----------
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print("测试 dropout_layer:")
    print(X)
    print("dropout=0:", dropout_layer(X, 0.))
    print("dropout=0.5:", dropout_layer(X, 0.5))
    print("dropout=1:", dropout_layer(X, 1.))

    # ---------- 定义模型参数 ----------
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    # ---------- 从零实现带Dropout的MLP ----------
    class Net(nn.Module):
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                     is_training=True):
            super(Net, self).__init__()
            self.num_inputs = num_inputs
            self.training = is_training
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()

        def forward(self, X):
            H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
            if self.training:
                H1 = dropout_layer(H1, dropout1)
            H2 = self.relu(self.lin2(H1))
            if self.training:
                H2 = dropout_layer(H2, dropout2)
            return self.lin3(H2)

    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    # ---------- 训练 ----------
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    print("\n>>> 从零实现Dropout——开始训练")
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    # ---------- 简洁实现 ----------
    net2 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net2.apply(init_weights)

    trainer2 = torch.optim.SGD(net2.parameters(), lr=lr)
    print("\n>>> 简洁实现Dropout——开始训练")
    d2l.train_ch3(net2, train_iter, test_iter, loss, num_epochs, trainer2)


# ============================================================
# 4.10  实战Kaggle比赛：预测房价
# ============================================================

def section_4_10_kaggle_house():
    """Kaggle房价预测：数据预处理 + K折交叉验证 + MLP模型"""
    print("=" * 50)
    print("4.10  实战Kaggle比赛：预测房价")
    print("=" * 50)

    import hashlib
    import os
    import numpy as np
    import pandas as pd
    import requests

    # ---------- 下载和缓存数据集 ----------
    DATA_HUB = dict()
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

    DATA_HUB['kaggle_house_train'] = (
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')

    DATA_HUB['kaggle_house_test'] = (
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    def download(name, cache_dir=os.path.join('..', 'data')):
        assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
        url, sha1_hash = DATA_HUB[name]
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.join(cache_dir, url.split('/')[-1])
        if os.path.exists(fname):
            sha1 = hashlib.sha1()
            with open(fname, 'rb') as f:
                while True:
                    data = f.read(1048576)
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                return fname
        print(f'正在从{url}下载{fname}...')
        r = requests.get(url, stream=True, verify=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return fname

    # ---------- 数据预处理 ----------
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # 标准化数值特征
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 独热编码
    all_features = pd.get_dummies(all_features, dummy_na=True)
    all_features = all_features.astype(float)  # 确保所有列都是数值类型

    # 转换为PyTorch张量
    n_train = train_data.shape[0]
    train_features = torch.tensor(
        all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(
        all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    # ---------- 模型 ----------
    loss = nn.MSELoss()
    in_features = train_features.shape[1]

    def get_net():
        net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1))
        return net

    def log_rmse(net, features, labels):
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
        return rmse.item()

    def train(net, train_features, train_labels, test_features, test_labels,
              num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [], []
        train_iter = d2l.load_array(
            (train_features, train_labels), batch_size)
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                optimizer.step()
            train_ls.append(log_rmse(net, train_features, train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        return train_ls, test_ls

    # ---------- K折交叉验证 ----------
    def get_k_fold_data(k, i, X, y):
        assert k > 1
        fold_size = X.shape[0] // k
        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid

    def k_fold(k, X_train, y_train, num_epochs, learning_rate,
               weight_decay, batch_size):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            data = get_k_fold_data(k, i, X_train, y_train)
            net = get_net()
            train_ls, valid_ls = train(net, *data, num_epochs,
                                       learning_rate, weight_decay, batch_size)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
                  f'验证log rmse{float(valid_ls[-1]):f}')
        return train_l_sum / k, valid_l_sum / k

    # ---------- 模型选择 ----------
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs,
                              lr, weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')

    # ---------- 最终训练并提交 ----------
    def train_and_pred(train_features, test_features, train_labels, test_data,
                       num_epochs, lr, weight_decay, batch_size):
        net = get_net()
        train_ls, _ = train(net, train_features, train_labels, None, None,
                            num_epochs, lr, weight_decay, batch_size)
        d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
                 ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
        print(f'训练log rmse：{float(train_ls[-1]):f}')
        preds = net(test_features).detach().numpy()
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat(
            [test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('submission.csv', index=False)
        print("预测结果已保存至 submission.csv")

    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)


# ============================================================
# ═══════════════  运行入口：想运行哪节就去掉注释  ═══════════════
# ============================================================

if __name__ == "__main__":
    # 每个函数独立运行，互不干扰。
    # 去掉下面任意一行的 # 即可运行对应小节：

    # section_4_2_mlp_scratch()      # 4.2  MLP从零实现 (Fashion-MNIST)
    # section_4_5_weight_decay()     # 4.5  权重衰减
    # section_4_6_dropout()          # 4.6  Dropout暂退法
    section_4_10_kaggle_house()    # 4.10 Kaggle房价预测

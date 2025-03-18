import torch
from my_train_accuracy import accuracy
from my_train_accuracy import Accumulator

def train_epoch_ch3(net, train_iter, loss, updater, pbar):
    """
    训练模型一个迭代周期
    输入：
    net: 神经网络
    train_iter: 训练数据集
    loss: 损失函数
    updater: 更新参数的函数
    输出：
    训练损失、训练准确度
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        if pbar:
            pbar.update(1)
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
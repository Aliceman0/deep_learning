import torch
from my_train_accuracy import accuracy
from my_train_accuracy import Accumulator


def evaluate_accuracy(net, data_iter):
    """
    计算模型在数据集上的准确率
    输入：
    net: 神经网络
    data_iter: 数据集
    输出：
    准确率
    """
    if isinstance(net, torch.nn.Module):
        net.eval() # 设置为评估模式
    metric = Accumulator(2) # 正确预测数，预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
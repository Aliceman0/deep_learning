import torch

class Accumulator:
    """
    在多个变量上累加
    输入：
    n: 变量数量
    *args: 变量
    idx: 变量索引
    输出:
    索引对应的变量
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def accuracy(y_hat, y):
    """
    计算预测正确的数量
    输入：
    y_hat: 预测值
    y: 真实值
    输出：
    预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # 如果预测结果是矩阵
        y_hat = y_hat.argmax(axis=1)    # one-hot转索引
    cmp = y_hat.type(y.dtype) == y      # 比较真实标签
    # return cmp.float().sum()
    return float(cmp.type(y.dtype).sum())   # 也可以这样写,之前报错了

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


if __name__ == "__main__":
    import torch
    y_hat = torch.tensor([1, 2])
    y = torch.tensor([0, 2])
    print(accuracy(y_hat, y) / len(y)) # 0.5
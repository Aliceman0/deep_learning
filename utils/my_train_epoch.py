from my_train_accuracy import accuracy
from my_train_accuracy import Accumulator
from my_test_accuracy import evaluate_accuracy
from my_train_func import train_epoch_ch3
from my_plot_acc import Animator
from tqdm import tqdm


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型函数
    输入：
    net: 神经网络
    train_iter: 训练数据集
    test_iter: 测试数据集
    loss: 损失函数
    num_epochs: 迭代周期数
    updater: 更新参数的函数
    输出：
    画图
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'valid acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print()
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc



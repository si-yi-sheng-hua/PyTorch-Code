# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# 定义LSTM神经网络模型
class LstmRNN(nn.Module):
    """
    参数说明：
    - input_size：输入特征的维度（每个时间点的特征个数）
    - hidden_size：隐藏层的单元数，即隐藏状态的维度
    - output_size：输出的特征维度（如预测的值）
    - num_layers：堆叠的LSTM层数
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        # 创建LSTM层，输入维度为input_size，隐藏层有hidden_size个单元，堆叠层数为num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 线性层，将LSTM的输出映射到最终的输出
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        # 前向传播流程
        # _x的形状为(seq_len, batch, input_size)
        x, _ = self.lstm(_x)  # 通过LSTM层，得到每个时间点的隐藏状态输出
        s, b, h = x.shape  # 分别是序列长度、批次大小、隐藏层单元数
        # 将输出展平，为线性层准备
        x = x.view(s * b, h)
        # 通过线性层得到预测值
        x = self.forwardCalculation(x)
        # 恢复原始序列形状
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':
    # 生成时间序列数据
    data_len = 200  # 数据点总数
    t = np.linspace(0, 12 * np.pi, data_len)  # 生成0到12π的等间距点
    sin_t = np.sin(t)  # 计算sin值
    cos_t = np.cos(t)  # 计算cos值

    # 构造数据集，将sin和cos作为特征
    dataset = np.zeros((data_len, 2))
    dataset[:, 0] = sin_t  # 第一列为sin值
    dataset[:, 1] = cos_t  # 第二列为cos值
    dataset = dataset.astype('float32')  # 转成float类型，为训练准备

    # 画出部分原始数据（sin和cos的变化）
    plt.figure()
    plt.plot(t[0:60], dataset[0:60, 0], label='sin(t)')
    plt.plot(t[0:60], dataset[0:60, 1], label='cos(t)')
    # 添加垂直分割线，标记位置
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5')  # t=2.5位置
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8')  # t=6.8位置
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')

    # 选择训练和测试数据的比例
    train_data_ratio = 0.5  # 训练数据占一半
    train_data_len = int(data_len * train_data_ratio)

    # 划分训练集（输入和目标）
    train_x = dataset[:train_data_len, 0]  # 训练输入：sin值
    train_y = dataset[:train_data_len, 1]  # 训练目标：cos值
    INPUT_FEATURES_NUM = 1  # 每个时间点的特征数（这里只是sin）
    OUTPUT_FEATURES_NUM = 1  # 输出特征（cos值）
    t_for_training = t[:train_data_len]

    # 测试集
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- 训练 -------------------
    # 将训练数据转换成适合LSTM输入的形状（批次大小，序列长度，特征数）
    # 这里设置每个批次的序列长度为5
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM)
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM)

    # 转成PyTorch的tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    # 创建LSTM模型实例，隐藏层有16个单元
    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1)
    print('LSTM模型结构:', lstm_model)
    print('模型参数:', lstm_model.parameters)

    # 定义损失函数（均方误差）和优化器
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    # 训练循环
    max_epochs = 10000
    for epoch in range(max_epochs):
        # 前向传播得到预测值
        output = lstm_model(train_x_tensor)
        # 计算损失值
        loss = loss_function(output, train_y_tensor)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 损失足够小时停止训练
        if loss.item() < 1e-4:
            print('第{}轮训练，损失: {:.5f}'.format(epoch + 1, loss.item()))
            print("损失已经达到目标值，总结训练")
            break
        elif (epoch + 1) % 100 == 0:
            print('第{}轮训练，损失: {:.5f}'.format(epoch + 1, loss.item()))

    # 训练完毕后，用训练好的模型预测训练集
    predictive_y_for_training = lstm_model(train_x_tensor)
    # 转成numpy数组用于绘图
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- 测试 -------------------
    # 也可以加载保存的模型参数进行测试（此处未用到）
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))
    lstm_model = lstm_model.eval()  # 转为测试模式，关闭dropout等
    # 测试集输入，要保持形状一致
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)

    # 进行测试集预测
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- 绘图显示结果 -------------------
    plt.figure()
    # 训练集实际sin、cos和预测
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')

    # 测试集实际sin、cos和预测
    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')

    # 分隔线，标识训练和测试部分
    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line')

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size=15, alpha=1.0)
    plt.text(20, 2, "test", size=15, alpha=1.0)

    plt.show()

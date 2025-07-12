#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt    # 用于绘图
import numpy as np                 # 数组处理
import tushare as ts              # 用于获取股票数据
import pandas as pd               # 处理数据
import torch                      # PyTorch深度学习库
from torch import nn             # 神经网络模块
import datetime                  # 日期时间处理
import time                      # 时间处理

# 设置训练的天数窗口（即用多少天数据预测下一天）
DAYS_FOR_TRAIN = 10

# 定义LSTM回归模型类
class LSTM_Regression(nn.Module):
    """
    使用LSTM进行时间序列预测（回归）
    参数：
    - input_size: 每个输入序列的特征数（这里是1，也就是价格）
    - hidden_size: LSTM隐藏层的神经元数量
    - output_size: 输出的维度（这里是1，预测一个值）
    - num_layers: 堆叠的LSTM层数（隐藏层的层数，一般为1或2）
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        # 初始化LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 线性层将隐藏状态映射到预测值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        # 输入：_x，形状 (序列长度, 批次, 特征数)
        # - 序列长度：表示序列中时间点的数量（这里是DAYS_FOR_TRAIN，比如10天）
        # - 批次：每次输入的样本数量（有多少个样本同时处理）
        # - 特征数：每个时间点的特征维度（这里默认为1，因为是价格）

        # 1.运行LSTM层
        # 这会输出两个内容：每个时间点的隐藏状态（x），以及最后一个隐藏状态（这里用不到）
        x, _ = self.lstm(_x)
        # x: 一个包含了序列中每个时间步隐藏状态的张量。这是序列模型的核心输出，后续会用它来预测
        # _: LSTM最后一个时间步的隐藏状态等信息
        # x的形状： (序列长度, 批次, 隐藏神经元个数)
        # 每个时间点（序列的每个元素）都对应一个隐藏状态

        # 解析形状
        s, b, h = x.shape
        # s：序列长度（比如10）
        # b：批次大小（比如一批中有多少样本）
        # h：隐藏层的神经元数（这里是8）

        # 2.改变x的形状
        # 以便将序列中所有时间点的隐藏状态合并成一个二维数组
        x = x.view(s * b, h)
        # 现在x的形状：(序列长度 * 批次, 隐藏神经元个数)
        # 这个操作会将所有时间点的隐藏状态堆叠在一起，准备送入线性层

        # 3.通过线性层（全连接层）进行变换
        x = self.fc(x)
        # 线性层输出的形状： (s * b, 输出维度)，这里输出维度为1
        # 每个时间点对应一个预测值

        # 4.恢复原始序列的形状
        x = x.view(s, b, -1)
        # 变回：(序列长度, 批次, 输出维度)
        # 这样可以保持序列的时间结构，方便后续处理（如序列长度一致的输出）

        # 返回最终预测值
        return x

# 创建数据集函数：从时间序列中生成输入输出对
def create_dataset(data, days_for_train=5):
    """
    生成训练用的数据集
    - data：一维序列（如价格）
    - days_for_train：每个输入序列的长度
    返回：
    - 数据集的输入（每个样本是连续days_for_train天的价格）
    - 对应的标签（下一天的价格）
    """
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train): #左闭右开
        _x = data[i:(i + days_for_train)]  # 取连续的days_for_train天
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])  # 下一天的值作为标签
        # 用从第i天到第i + days_for_train - 1天的历史数据，预测第i + days_for_train天的值。
    return np.array(dataset_x), np.array(dataset_y)

if __name__ == '__main__':
    t0 = time.time()  # 记录开始时间

    # 获取上证指数的历史收盘价，起始日期为2019-01-01
    data_close = ts.get_k_data('000001', start='2019-01-01', index=True)['close']
    data_close.to_csv('000001.csv', index=False)  # 保存为csv文件
    data_close = pd.read_csv('000001.csv')        # 读入数据

    # 备用：获取上海证券交易所指数（可选）
    df_sh = ts.get_k_data('sh', start='2019-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))
    print(df_sh.shape)

    # 转换数据类型为float32（浮点数）
    data_close = data_close.astype('float32').values

    # 绘制原始价格走势
    plt.plot(data_close)
    plt.savefig('data.png', format='png', dpi=200)  # 保存图片
    plt.close()

    # 归一化：将价格缩放到0-1区间，有助于模型训练
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)

    # 使用create_dataset生成训练数据
    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)
    # 这里：dataset_x.shape = (样本数量, 10), 因为用10天数据预测下一天

    # 划分训练集和测试集（70%训练，30%测试）
    train_size = int(len(dataset_x) * 0.7)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]

    # 改变训练数据的形状以适应PyTorch的LSTM输入
    # 形状：(序列长度, 批次, 特征数)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # 变成 (样本数, 1, 10)
    train_y = train_y.reshape(-1, 1, 1)               # 变成 (样本数, 1, 1)

    # 转化为PyTorch的tensor
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    # 1.初始化模型
    model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)

    # 计算模型参数（了解模型大小）
    model_total = sum([param.nelement() for param in model.parameters()])
    print("Number of model_total parameter: %.8fM" % (model_total / 1e6))

    # 2.训练准备
    train_loss = []
    loss_function = nn.MSELoss()  # 均方误差损失，用于回归问题
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Adam优化器

    # 3.开始训练（200轮）
    for i in range(200):
        out = model(train_x)  # 模型输出
        loss = loss_function(out, train_y)  # 计算预测误差
        loss.backward()      # 反向传播
        optimizer.step()     # 更新参数
        optimizer.zero_grad()# 清空梯度
        train_loss.append(loss.item())  # 记录损失值

        # 记录训练信息到文件
        with open('log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i + 1, loss.item()))
        if (i + 1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    # 绘制训练损失曲线，观察模型收敛情况
    plt.figure()
    plt.plot(train_loss, 'b', label='loss')
    plt.title("Train_Loss_Curve")
    plt.ylabel('train_loss')
    plt.xlabel('epoch_num')
    plt.savefig('loss.png', format='png', dpi=200)
    plt.close()

    # 保存模型参数（可选，用于以后加载）
    # torch.save(model.state_dict(), 'model_params.pkl')

    t1 = time.time()  # 记录结束时间
    T = t1 - t0        # 计算训练耗时
    print('The training time took %.2f' % (T / 60) + ' mins.')

    # 输出起始和结束时间
    tt0 = time.asctime(time.localtime(t0))
    tt1 = time.asctime(time.localtime(t1))
    print('The starting time was ', tt0)
    print('The finishing time was ', tt1)

    # 测试：用训练集上的所有数据做预测（评估模型）
    model = model.eval()  # 转为评估模式（关闭dropout等）
    # 如果有保存的模型参数，可以加载
    # model.load_state_dict(torch.load('model_params.pkl'))

    # 生成完整数据集的预测
    dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    dataset_x = torch.from_numpy(dataset_x)

    pred_test = model(dataset_x)  # 全部训练集输入模型预测
    pred_test = pred_test.view(-1).data.numpy()  # 转为numpy数组
    # 为了匹配原始数据长度，前面填充0
    pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))
    assert len(pred_test) == len(data_close)

    # 绘制真实值和预测值对比图
    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(data_close, 'b', label='real')
    plt.plot((train_size, train_size), (0, 1), 'g--')  # 训练与测试分界线
    plt.legend(loc='best')
    plt.savefig('result.png', format='png', dpi=200)
    plt.close()

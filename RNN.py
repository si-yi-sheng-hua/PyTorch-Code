import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 数据集：字符序列预测（Hello -> Elloh）
# 目标是训练一个RNN，将 "hello" 转换为 "elloh"
char_set = list("hello")  # 定义字符集
char_to_idx = {c: i for i, c in enumerate(char_set)}  # 字符到索引的映射：{'h': 0, 'e': 1, 'l': 3, 'o': 4}
idx_to_char = {i: c for i, c in enumerate(char_set)}  # 索引到字符的映射：{0: 'h', 1: 'e', 2: 'l', 3: 'l', 4: 'o'}

# 数据准备
input_str = "hello"  # 输入字符串
target_str = "elloh"  # 目标字符串
input_data = [char_to_idx[c] for c in input_str]  # 将输入字符串转换为索引列表：[0, 1, 3, 3, 4]
target_data = [char_to_idx[c] for c in target_str]  # 将目标字符串转换为索引列表：[1, 3, 3, 4, 0]

# 转换为独热编码 (One-Hot Encoding)
input_one_hot = np.eye(len(char_set))[input_data]  # 创建独热编码矩阵,  np.eye 生成单位矩阵，然后根据input_data提取对应的行
#input_one_hot[[1. 0. 0. 0. 0.], [0. 1. 0. 0. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 1. 0.], [0. 0. 0. 0. 1.]]

# 转换为 PyTorch Tensor
inputs = torch.tensor(input_one_hot, dtype=torch.float32)  # 将独热编码转换为PyTorch张量，数据类型为float32
targets = torch.tensor(target_data, dtype=torch.long)  # 将目标索引转换为PyTorch张量，数据类型为long
# inputs:tensor([[1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 0., 0., 0., 1.]])
# targets:tensor([1, 3, 3, 4, 0])

# 模型超参数
input_size = len(char_set)  # 输入特征的维度 (字符集大小，这里是5)
hidden_size = 8  # 隐藏层的大小 (超参数，可调整)
output_size = len(char_set)  # 输出特征的维度 (字符集大小，这里是5)
num_epochs = 100  # 训练的轮数
learning_rate = 0.1  # 学习率

# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        # 一个隐藏层: 你可以理解这是一个包含多个神经元的层，它位于输入层和输出层之间。
        # 8个神经元(隐藏单元): 这个隐藏层包含了hidden_size=8个神经元( or 隐藏单元)。
        # 每个神经元接收来自前一层的输入（对于RNN，输入可能是之前时间步的隐藏状态和当前时间步的输入），
        # 并将其输出传递给下一层(要么是输出层，要么是下一个时间步的同一隐藏层)。
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # 定义RNN层，batch_first=True表示输入张量的第一个维度是batch_size
        self.fc = nn.Linear(hidden_size, output_size)  # 定义一个全连接层，将RNN的输出映射到输出的维度

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)  # 前向传播，x是输入，hidden是隐藏状态
        out = self.fc(out)  # 应用全连接层
        return out, hidden  # 返回输出和隐藏状态

model = RNNModel(input_size, hidden_size, output_size)  # 实例化RNN模型

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器，学习率设置为learning_rate

# 训练 RNN
losses = []
hidden = None  # 初始隐藏状态为 None
for epoch in range(num_epochs):
    # 1. 梯度清零
    optimizer.zero_grad()
    # 2. 前向传播
    #   - inputs.unsqueeze(0):  将输入张量inputs的形状从 (5,5)变为(1, 5, 5)，以添加batch维度
    inputs_unsqueeze = inputs.unsqueeze(0)
    # tensor([[[1., 0., 0., 0., 0.],
    #          [0., 1., 0., 0., 0.],
    #          [0., 0., 0., 1., 0.],
    #          [0., 0., 0., 1., 0.],
    #          [0., 0., 0., 0., 1.]]]),unsqueeze后最外层多了一个[]
    #   - model(inputs.unsqueeze(0), hidden): 将经过独热编码后的输入数据和隐藏状态传递到模型中，得到输出和新的隐藏状态。
    #   - out: (1, 5, 5): 由于batch_first=True，输出形状是(batch_size, sequence_length, hidden_size )，其中batch_size=1, sequence_length=5, hidden_size=5 (output_size)。
    #   - hidden: (1, 5, 8) : 经过RNN层后的隐藏状态
    outputs, hidden = model(inputs_unsqueeze, hidden)
    #outputs(epoch:0):
    # tensor([[[ 0.1112,  0.0515,  0.2828,  0.0069, -0.1267],
    #          [ 0.0043,  0.2659,  0.4750, -0.0836, -0.1345],
    #          [-0.0...  0.0269,  0.2292,  0.1497, -0.1357],
    #          [-0.1298,  0.0620,  0.3718,  0.0901, -0.0947]]],
    #        grad_fn=<ViewBackward0>)
    # hidden(epoch:0):
    # tensor([[[ 0.2681, -0.1092,  0.4549, -0.5249, -0.5854, -0.2856, -0.3214,
    #            0.8051]]], grad_fn=<StackBackward0>)

    # 3. 分离之前的隐藏状态(防止梯度爆炸)
    #  -   hidden.detach():
    # 	在每一次迭代中，将旧的隐藏状态从计算图中分离出来，`detach()` 方法是用于分离张量与其创建历史的。
    #   这样`hidden`就不会与之前的计算图连接，从而减少了梯度反向传播的长度，确保梯度不会在多个迭代中累积，使得每个epoch都是新的开始。
    hidden = hidden.detach()  # 防止梯度爆炸

    # 4. 计算损失:
    #  - outputs.view(-1, output_size):  outputs的形状是(1, 5, 5)，在这里将其变形为(25, 5)，即(sequence_length * batch_size, hidden_size)
    #  - targets:  targets的形状是(5)，targets的形状是(sequence_length*)
    #  - criterion(outputs.view(-1, output_size), targets): CrossEntropyLoss期待的输入形状是(batch_size, num_classes) 和 (batch_size)，
    #     因此需要调整输出的形状。loss值，单个标量
    # outputs.view(-1,output_size):tensor([[ 0.4883, -0.5030,  0.1942, -0.1366, -0.2441],
    #         [ 0.5302, -0.5007,  0.0310, -0.0636,  0.0434],
    #         [ 0.4073...7, -0.4466,  0.1047, -0.1732, -0.1509],
    #         [ 0.6676, -0.4704,  0.2150, -0.1781, -0.1448]],
    #        grad_fn=<ViewBackward0>)
    loss = criterion(outputs.view(-1,output_size), targets)

    # 5. 反向传播:
    #  - loss.backward():计算损失对于所有模型参数的梯度，也就是计算损失函数对模型的各个参数的偏导数
    loss.backward()

    # 6. 优化器更新参数
    #  - optimizer.step():根据计算得到的梯度，更新模型的参数。使用梯度下降方法，朝着降低损失函数的方向更新参数
    optimizer.step()
    losses.append(loss.item())

    # 7. 打印信息
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# 测试 RNN
with torch.no_grad():  # 禁用梯度计算，用于测试
    test_hidden = None
    test_output, _ = model(inputs.unsqueeze(0), test_hidden) #前向传播产生输出。  这里输入的inputs，也需要增加一个batch维度。
    predicted = torch.argmax(test_output, dim=2).squeeze().numpy()  # 获取预测结果，argmax找到每个时间步预测概率最高的字符的索引， dim=2意味着沿着最后一个维度。squeeze() 去掉batch_size维度，只剩下一个序列。  .numpy()将结果转换为NumPy数组

    print("Input sequence: ", ''.join([idx_to_char[i] for i in input_data]))  # 打印输入序列
    print("Predicted sequence: ", ''.join([idx_to_char[i] for i in predicted]))  # 打印预测序列

# 可视化损失
plt.plot(losses, label="Training Loss")  # 绘制损失曲线
plt.xlabel("Epoch")  # x轴标签
plt.ylabel("Loss")  # y轴标签
plt.title("RNN Training Loss Over Epochs")  # 图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

import torch
import numpy as np
import matplotlib.pyplot as plt

# 随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 生成训练数据
X = torch.randn(100, 2)  # 100 个样本，每个样本 2 个特征
true_w = torch.tensor([2.0, 3.0])  # 假设真实权重
true_b = 4.0  # 偏置项
Y = X @ true_w + true_b + torch.randn(100) * 0.1  # 加入一些噪声

# 打印部分数据
print(X[:5])
print(Y[:5])


import torch.nn as nn

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入为2个特征，输出为1个预测值
        self.linear = nn.Linear(2, 1)  # 输入维度2，输出维度1

    def forward(self, x):
        return self.linear(x)  # 前向传播，返回预测结果


# 创建模型实例
model = LinearRegressionModel()

# 损失函数（均方误差）
criterion = nn.MSELoss()

# 优化器（使用 SGD 或 Adam）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率设置为0.01

# optimizer 通过 model.parameters() 获取模型所有可学习的参数，
# 这样，优化器就“知道”要更新哪些参数，这实际上就是模型的参数。
# criterion 通常不与模型绑定，只需要定义一个损失函数实例（例如 nn.MSELoss()），
# 然后在训练时用模型输出和标签计算损失。
# loss.backward() 和 optimizer.step() 是两个独立的调用，
# 但它们之间通过 PyTorch的计算图（Computation Graph） 和 张量的 .grad 连接在一起。

# 训练模型
num_epochs = 500  # 训练 500 轮
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    # 前向传播：数据 X 经过模型 model 的层层运算，生成预测值 predictions。
    # 这一步构建了从输入到输出的计算图。
    # 这个图记录了所有张量（X，模型的权重，偏置，中间激活，predictions）以及它们之间的运算关系。
    predictions = model(X)  # 模型输出预测值

    # 计算损失：根据预测值 predictions 和真实标签 Y 计算损失 loss。
    # loss张量也被添加到计算图的末端。loss是由predictions（以及Y）通过criterion计算得出的。
    # 损失函数也是计算图的一部分，loss 是整个图的标量输出。
    # 进行.squeeze()操作的主要目的是确保predictions和Y的维度是匹配的，以符合损失函数的输入要求。
    loss = criterion(predictions.squeeze(), Y)  # 计算损失（注意预测值需要压缩为1D）

    # 反向传播：
    optimizer.zero_grad()  # 清空之前批次的梯度，防止梯度累积。
                           # 例如，对于一个参数param其param.grad会被设置为None或零张量。

    loss.backward()  # 从损失 loss 开始，沿着之前构建的计算图反向传播。
                     # 这一步利用链式法则计算 loss 对模型中所有可训练参数的梯度，
                     # 并将这些梯度存储在对应参数的 .grad 属性中。
                     # 例如，如果有一个参数 W，那么 loss.backward() 会计算 d(loss)/d(W)，
                     # 并将其赋值给 W.grad。

    optimizer.step()  # 更新模型参数。
                      # 计算图的视角补充：
                      # 优化器（optimizer）会遍历模型中所有可训练的参数
                      # 这些参数通常已经在其.grad属性中包含了通过loss.backward()计算得到的梯度。
                      # 然后，根据选定的优化算法（如 SGD, Adam 等），使用这些梯度来更新参数的值。
                      # 例如，对于 SGD，参数更新的公式通常是：param = param - learning_rate * param.grad。
                      # optimizer.step() 不再涉及计算图的构建或梯度计算，它仅是基于已计算出的梯度来修改参数的值。


    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}')

# 查看训练后的权重和偏置
print(f'Predicted weight: {model.linear.weight.data.numpy()}')
print(f'Predicted bias: {model.linear.bias.data.numpy()}')

# 在新数据上做预测
with torch.no_grad():  # 评估时不需要计算梯度
    predictions = model(X)

# 可视化预测与实际值
plt.scatter(X[:, 0], Y, color='blue', label='True values')
plt.scatter(X[:, 0], predictions, color='red', label='Predictions')
plt.legend()
plt.show()
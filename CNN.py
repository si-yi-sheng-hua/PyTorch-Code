import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层：输入1通道，输出32通道，卷积核大小3x3
        # conv1 是第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels=1,     # 输入通道数：
                               # 对于灰度图像，通常是1。
                               # 对于RGB彩色图像，通常是3。
                               # pytorch的卷积层预期输入形状为 (batch_size, channels, height, width)。
            out_channels=32,   # 输出通道数：
                               # 这个卷积层会学习32个不同的特征检测器（滤波器）。
                               # 每个滤波器会生成一个特征图，因此输出会有32个特征图。
            kernel_size=3,     # 卷积核大小：
                               # 这是一个 3x3 的卷积核。
                               # (如果传入一个整数 N，则表示核大小为 NxN；也可以传入一个元组 (H, W))。
            stride=1,          # 步长：
                               # 卷积核在输入特征图上每次移动的像素数。
                               # stride=1 表示卷积核每次移动1个像素。
                               # (如果传入一个整数 S，则表示水平和垂直步长均为 S；也可以传入一个元组 (SH, SW))。
            padding=1          # 填充：
                               # 在输入特征图的边界周围添加的零像素数量。
                               # padding=1 表示在输入特征图的每一侧（上、下、左、右）都添加1圈零像素。
                               # 这通常用于保持输出特征图的 H 和 W 与输入大致相同 (当 stride=1 时)。
                               # 计算公式：输出边长 = (输入边长 - 卷积核边长 + 2 * padding) / stride + 1
                               # 如果输入图像是 28x28，那么经过 conv1 (kernel=3, stride=1, padding=1) 后，
                               # 输出特征图的尺寸仍将是 28x28。
        )

        # 定义卷积层：输入32通道，输出64通道
        # conv2 是第二个卷积层
        self.conv2 = nn.Conv2d(
            in_channels=32,    # 输入通道数：
                               # 它的输入是前一个卷积层 (conv1) 的输出，conv1 的 out_channels 是 32。
            out_channels=64,   # 输出通道数：
                               # 这个卷积层会学习64个不同的特征检测器。
                               # 相比conv1，学习的特征可能更高级。
            kernel_size=3,     # 卷积核大小：依然是 3x3。
            stride=1,          # 步长：依然是1。
            padding=1          # 填充：依然是1。
                               # 经过 conv2 (kernel=3, stride=1, padding=1) 后，
                               # 如果输入是 14x14 (考虑池化后的尺寸)，输出仍将是 14x14。
        )

        # in_channels=32: conv2 层接收 32 个特征图的集合，可以将其想象成一个深度为 32 的三维张量，
        # 例如，前一层 conv1 的输出 output_conv1 的形状可能是 (batch_size, 32, H, W)。

        # out_channels=64: conv2 层将产生 64 个新的特征图作为输出。

        # 为了生成 conv2 的第一个输出特征图 (output_conv2_channel1)，会使用一个三维的卷积核，
        # 形状为 (in_channels, kernel_height, kernel_width)，在这里就是 (32, 3, 3)。
        # 这个 (32, 3, 3) 的卷积核会同时对 conv2 的所有 32 个输入通道进行卷积。

        # 具体来说，这个 (32, 3, 3) 的卷积核会：
        #   1. 对 input_conv2 的第一个通道进行 3x3 卷积。
        #   2. 对 input_conv2 的第二个通道进行 3x3 卷积。
        #   ...
        #   3. 最后，将这 32 次卷积的结果在逐元素相加，再加上一个偏置项，从而得到 conv2 的第一个输出特征图 (output_conv2_channel1)。
        #      这整个过程只产生了一个输出特征图。

        # conv2 层需要生成 out_channels=64 个输出特征图。
        # 因此，conv2 会有 64 个独立的三维卷积核，每个卷积核的形状都是 (32, 3, 3)。
        # 每个这样的 (32, 3, 3) 卷积核都会按照上述方式 (对所有 32 个输入通道进行卷积，然后求和) 生成一个输出特征图。

        # 定义全连接层
        # fc1 是第一个全连接层 (也称为线性层、密集层)
        # 64 * 7 * 7 是根据网络配置推断出的特征图展平后的尺寸
        # 计算依据：
        # 假设输入图像是 28x28 (MNIST/FashionMNIST常用尺寸)
        # 1. 经过 conv1 (kernel=3, stride=1, padding=1)：输出 32x28x28
        # 2. 经过 F.max_pool2d (kernel_size=2)：输出 32x14x14 (因为 28 / 2 = 14)
        # 3. 经过 conv2 (kernel=3, stride=1, padding=1)：输出 64x14x14
        # 4. 经过 F.max_pool2d (kernel_size=2)：输出 64x7x7 (因为 14 / 2 = 7)
        # 展平操作 x.view(-1, 64 * 7 * 7) 会将 64x7x7 的特征图展平成一个长向量。
        # 因此，输入特征的数量是 64 * 7 * 7。
        self.fc1 = nn.Linear(
            in_features=64 * 7 * 7, # 输入特征的数量：
                                    # 这是展平后的特征图的维度。
                                    # 64是第二个卷积层 (conv2) 的输出通道数。
                                    # 7x7 是经过两次最大池化后，每个特征图的空间尺寸。
            out_features=128        # 输出特征的数量：
                                    # 这个全连接层将原始的特征向量映射到128维的向量空间。
                                    # 这是隐藏层的大小，一个超参数，可根据模型复杂度调整。
        )

        # fc2 是第二个（也是最后一个）全连接层
        self.fc2 = nn.Linear(
            in_features=128,  # 输入特征的数量：
                              # 它的输入是前一个全连接层 (fc1) 的输出，fc1 的 out_features 是 128。
            out_features=10   # 输出特征的数量：
                              # 这是模型的最终输出维度，通常等于分类任务的类别数量。
                              # 对于MNIST手写数字识别，有0-9共10个类别。
        )

    def forward(self, x):
        # forward 方法定义了数据在网络中流动的路径

        # x 的初始形状通常是 (batch_size, 1, H, W)
        # 例如，对于MNIST，可能是 (64, 1, 28, 28)

        # 第一层卷积 + ReLU 激活
        # conv1 的输出形状：(batch_size, 32, H, W) --> (batch_size, 32, 28, 28)
        x = F.relu(self.conv1(x))

        # 最大池化层：缩小特征图的空间尺寸
        # kernel_size=2 表示 2x2 的池化窗口。
        # 默认 stride=kernel_size，即步长也是2，所以池化操作会将 H 和 W 减半。
        # 池化后的形状：(batch_size, 32, H/2, W/2) --> (batch_size, 32, 14, 14)
        x = F.max_pool2d(x, 2)

        # 第二层卷积 + ReLU 激活
        # conv2 的输入形状：(batch_size, 32, 14, 14)
        # conv2 的输出形状：(batch_size, 64, 14, 14)
        x = F.relu(self.conv2(x))

        # 再次最大池化层：进一步缩小特征图的空间尺寸
        # 池化后的形状：(batch_size, 64, H/4, W/4) --> (batch_size, 64, 7, 7)
        x = F.max_pool2d(x, 2)

        # 展平操作 (Flatten): 将多维特征图展平为一维向量，以便输入全连接层
        # x.size(0) 获取批次大小 (batch_size)
        # -1 告诉 PyTorch 自动推断该维度的大小，这里就是指 64 * 7 * 7
        # 展平后的形状：(batch_size, 64 * 7 * 7) --> (batch_size, 3136)
        x = x.view(-1, 64 * 7 * 7)

        # 第一个全连接层 + ReLU 激活
        # fc1 的输入形状：(batch_size, 3136)
        # fc1 的输出形状：(batch_size, 128)
        x = F.relu(self.fc1(x))

        # 第二个全连接层 (输出层)
        # fc2 的输入形状：(batch_size, 128)
        # fc2 的输出形状：(batch_size, 10)
        # 注意：这里没有使用激活函数 (如 softmax)，因为在交叉熵损失函数 (nn.CrossEntropyLoss) 内部通常会包含 softmax 或 log_softmax 操作。
        x = self.fc2(x)

        return x

# 创建模型实例
model = SimpleCNN()

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. 模型训练
num_epochs = 5
model.train()  # 设置模型为训练模式

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 5. 模型测试
model.eval()  # 设置模型为评估模式
correct = 0
total = 0

with torch.no_grad():  # 关闭梯度计算
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 6. 可视化测试结果
dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predictions = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    axes[i].imshow(images[i][0], cmap='gray')
    axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
    axes[i].axis('off')
plt.show()
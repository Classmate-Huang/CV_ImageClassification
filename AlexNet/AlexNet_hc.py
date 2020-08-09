import torch.nn as nn


class LRN(nn.Module):       # Local Response Normalization
    def __init__(self, k=1.0, n=1, alpha=1.0, beta=0.75, ACROSS_CHANNLES=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNLES = ACROSS_CHANNLES
        if self.ACROSS_CHANNLES:
            self.average = nn.AvgPool3d(kernel_size=(n, 1, 1), stride=1,
                                        padding=(int(n-1.0)/2))
        else:
            self.average = nn.AvgPool2d(kernel_size=n, stride=1,
                                        padding=int((n-1.0)/2))
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def floward(self, a):
        if self.ACROSS_CHANNLES:
            div = a.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = (div * self.alpha + self.k).pow(self.beta)
        else:
            div = a.pow(2)
            div = self.average(div)
            div = (div * self.alpha + self.k).pow(self.beta)
        b = a / div
        return b


class AlexNet(nn.Module):
    def __init__(self, num_class=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(      # Convolutional 特征提取部分
            # Layer 1     input: 227,3     output: 27,96
            nn.Conv2d(3, 96, 11, stride=4, padding=0),  # 55
            nn.ReLU(inplace=True),
            LRN(n=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(3, stride=2),  # 27
            # Layer 2       input: 27,96     output: 13,256
            # group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
            nn.Conv2d(96, 256, 5, padding=2, groups=2),     # 27
            nn.ReLU(inplace=True),
            LRN(n=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(3, stride=2),  # 13
            # Layer 3       input: 13,256   output: 13,384
            nn.Conv2d(256, 384, 3, padding=1),  # 13  #
            nn.ReLU(inplace=True),
            # Layer 4       input: 13,384   output: 13,384
            nn.Conv2d(384, 384, 3, padding=1),  # 13
            nn.ReLU(inplace=True),
            # Layer 5       input: 13,384   output: 6,256
            nn.Conv2d(384, 256, 3, padding=1),  # 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )
        self.classifier = nn.Sequential(  # Classifier 分类器
            # Layer 6
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # Layer 7
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # Layer 8
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256*6*6)
        z = self.classifier(x)
        return z


# 查看网络结构
model = AlexNet(1000)
print(model)

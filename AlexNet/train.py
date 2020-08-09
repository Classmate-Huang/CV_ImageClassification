import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from AlexNet_hc import AlexNet


transform = transforms.Compose(
    [transforms.RandomSizedCrop(227),    # AlexNet输入尺寸
     transforms.RandomHorizontalFlip(),  # 随机翻转
     transforms.ToTensor(),  # 转为向量
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]  # 归一化
)

# 加载训练集 由于ImageNet数据集过于庞大150+GB，这里就没有进行下载了，有需要的可以去ImageNet官网下载
trainset = torchvision.datasets.ImageFolder('./the path to the ImageNet Data/../train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,  shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder('./the path to the ImageNet Data/../train', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,  shuffle=False, num_workers=0)


net = AlexNet(1000)

# 参数初始化
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0, std=0.01)
        nn.init.constant_(m.bias.data, 1)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.01)
        nn.init.constant_(m.bias.data, 1)

# for param in net.named_parameters():
#     print(param[0], param[1])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Loss Function & Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)

# 训练 Training
for epoch in range(1000):
    net.train()
    running_loss = 0.
    for i, data in enumerate(trainloader):
        img, label = data.to(device)
        # 清除缓存区
        optimizer.zero_grad()
        # 经典四步
        z = net(img)
        loss = criterion(z, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print("epoch %d, iter %d : %.3f" % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.
    path = 'model/AlexNet.pth'
    torch.save(net.state_dict(), path)

print('Finish train!')

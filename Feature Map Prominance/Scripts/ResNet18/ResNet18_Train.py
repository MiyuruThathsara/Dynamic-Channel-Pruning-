import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='../../../../data', train=True, transform= transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 0)

resnet18 = models.resnet18(pretrained=False)
resnet18 = resnet18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(20):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if( i%20 == 19 ):
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0
    PATH = './resnet18_epoch_%2d.pth' % ( epoch + 1 )
    torch.save(resnet18, PATH)

print('Finished Training!!!!!')
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from Custom_Net import Custom_Net as Net 

device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
print( "Device : ", device )

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR100(root = '../../../../data', train = True, download = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 0)
# testset = torchvision.datasets.CIFAR100(root = '../../../data', train = False, download = False, transform = transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 0)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.8)

print(dir(optimizer))

for epoch in range(200):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if(i%2000 == 1999):
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/2000))
            running_loss = 0

print(' Finished Training!!!!!!!! ')

PATH = './custom_net_CIFAR100_200_epochs.pth'
torch.save(net, PATH)
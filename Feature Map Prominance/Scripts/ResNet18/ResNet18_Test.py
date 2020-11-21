import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = '../../../data', train = False, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 0)

PATH = './resnet18_epoch_14.pth'
net = torch.load(PATH)
net.to(device)
print('Testing Started!!!!!!!!!!!!!')

# correct = 0
# total = 0

# with torch.no_grad():
#     for i,data in enumerate(testloader,1):
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         print('Accuracy = ', (predicted == labels).sum().item()/labels.size(0))
#         if(i%200 == 0):
#             print('Mini-Batch Count: %4d' % i)
#     print('Finished Testing!!!!!')


# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

correct = 0.0
total = 0.0
tot_map_drop = 0

with torch.no_grad():
    for i, data in enumerate(testloader, 1):
        images, labels = data[0].to(device), data[1].to(device)
        # layer = list(net._modules.items())[0]
        # layer[1].register_forward_hook(forward_hook)
        outputs, count, map_count = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        tot_map_drop += count/map_count
        correct += (predicted == labels).sum().item()
        print('Count = ', count, 'Map Count = ', map_count, 'Accuracy = ', (predicted == labels).sum().item()/labels.size(0) )
        # print(((predicted == labels).sum().item())/labels.size(0))
        if(i%200 == 0):
            print('Mini-Batch Count: %4d' % i)
    print('Finished Testing!!!!!!')
    print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
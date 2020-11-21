import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from ResNet50 import *

tot_count = 0
tot_map_count = 0

def forward_hook_pre( self, input ):
    global tot_count
    global tot_map_count
    count = 0
    map_count = 0
    for j,im in enumerate(input[0],0):
        im_size = im.size()[1] * im.size()[2]
        var = im.var(dim=(1,2))
        map_count += im.size()[0]
        max_var = max(var)
        input[0][j][ var <= (0.035 * max_var) ] = 0
        count += var[ var <= 0.035 * max_var ].size(0)
    tot_count += count
    tot_map_count += map_count

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = '../../../../data', train = False, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 0)

PATH = './resnet50_epoch_20.pth'
resnet50 = torch.load(PATH)
resnet50.eval()

layers = list(resnet50._modules.items())

layers[2][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[2][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[2][1][0].conv3.register_forward_pre_hook(forward_hook_pre)
layers[2][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[2][1][1].conv2.register_forward_pre_hook(forward_hook_pre)
layers[2][1][1].conv3.register_forward_pre_hook(forward_hook_pre)
layers[2][1][2].conv1.register_forward_pre_hook(forward_hook_pre)
layers[2][1][2].conv2.register_forward_pre_hook(forward_hook_pre)
layers[2][1][2].conv3.register_forward_pre_hook(forward_hook_pre)
layers[3][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[3][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[3][1][0].conv3.register_forward_pre_hook(forward_hook_pre)
layers[3][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[3][1][1].conv2.register_forward_pre_hook(forward_hook_pre)
layers[3][1][1].conv3.register_forward_pre_hook(forward_hook_pre)
layers[3][1][2].conv1.register_forward_pre_hook(forward_hook_pre)
layers[3][1][2].conv2.register_forward_pre_hook(forward_hook_pre)
layers[3][1][2].conv3.register_forward_pre_hook(forward_hook_pre)
layers[3][1][3].conv1.register_forward_pre_hook(forward_hook_pre)
layers[3][1][3].conv2.register_forward_pre_hook(forward_hook_pre)
layers[3][1][3].conv3.register_forward_pre_hook(forward_hook_pre)
layers[4][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][0].conv3.register_forward_pre_hook(forward_hook_pre)
layers[4][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][1].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][1].conv3.register_forward_pre_hook(forward_hook_pre)
layers[4][1][2].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][2].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][2].conv3.register_forward_pre_hook(forward_hook_pre)
layers[4][1][3].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][3].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][3].conv3.register_forward_pre_hook(forward_hook_pre)
layers[4][1][4].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][4].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][4].conv3.register_forward_pre_hook(forward_hook_pre)
layers[4][1][5].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][5].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][5].conv3.register_forward_pre_hook(forward_hook_pre)
layers[5][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[5][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[5][1][0].conv3.register_forward_pre_hook(forward_hook_pre)
layers[5][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[5][1][1].conv2.register_forward_pre_hook(forward_hook_pre)
layers[5][1][1].conv3.register_forward_pre_hook(forward_hook_pre)
layers[5][1][2].conv1.register_forward_pre_hook(forward_hook_pre)
layers[5][1][2].conv2.register_forward_pre_hook(forward_hook_pre)
layers[5][1][2].conv3.register_forward_pre_hook(forward_hook_pre)

resnet50 = resnet50.to(device)

correct = 0.0
total = 0.0
tot_map_drop = 0

print("Testing Has Started!!!!!!!")

with torch.no_grad():
    for i, data in enumerate(testloader, 1):
        tot_count = 0
        tot_map_count = 0
        images, labels = data[0].to(device), data[1].to(device)
        outputs = resnet50.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        tot_map_drop += tot_count/tot_map_count
        correct += (predicted == labels).sum().item()
        print('Count = ', tot_count, 'Map Count = ', tot_map_count, 'Accuracy = ', (predicted == labels).sum().item()/labels.size(0) )
        # print(((predicted == labels).sum().item())/labels.size(0))
        if(i%200 == 0):
            print('Mini-Batch Count: %4d' % i)
    print('Finished Testing!!!!!!')
    print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
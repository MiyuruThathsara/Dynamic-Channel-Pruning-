import matplotlib.pyplot as plt 
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
import torchvision.transforms as transforms
from ResNet18 import *

channel_list = []
def forward_pre_hook(self, input):
    global channel_list
    var_list = input[0].var(dim=(2,3))
    channel_list.append(var_list)

num_of_cal_imgs = 10000
num_of_conv_layers = 17
data_root = '../../../../data'
csv_filename = data_root + '/Calibration_CIFAR10.csv'
batch_size = 1

##################################################################################################################
# For Calibaration Dataset, dataloader
# transform_cal = transforms.Compose([ToTensor(), Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# cifar10_cal = CIFAR10_Calibration(csv_filename, transform=transform_cal)
# cal_dataloader = torch.utils.data.DataLoader(cifar10_cal, batch_size = batch_size, shuffle = False, num_workers = 0)
##################################################################################################################
# For Test Dataset, dataloader
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = data_root, train = False, download = False, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 0)
##################################################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

PATH = './resnet18_epoch_14.pth'
net = ResNet18()
model = torch.load(PATH)
net.load_state_dict(model.state_dict())
net.eval()

for name,module in net.named_modules():
    if(isinstance(module, nn.Conv2d)):
        if(name != 'conv1' and name != 'layer2.0.shortcut.0' and name != 'layer3.0.shortcut.0' and name != 'layer4.0.shortcut.0'):
            module.register_forward_pre_hook(forward_pre_hook)

dataiter = iter(testloader)
data = dataiter.next()
images, labels = data[0], data[1]
with torch.no_grad():
    outputs = net(images)
    plt.figure()
    for i in range(len(channel_list)):
        channel = channel_list[i].numpy()
        x = np.arange(0, channel[0].size, 1)
        plt.plot(x, channel[0])
    plt.legend(["conv1_0_1","conv1_0_2","conv1_1_1","conv1_1_2","conv2_0_1","conv2_0_2","conv2_1_1","conv2_1_2","conv3_0_1","conv3_0_2","conv3_1_1","conv3_1_2","conv4_0_1","conv4_0_2","conv4_1_1","conv4_1_2"])
    # plt.show()
    plt.savefig('ResNet18_Channel_Variance.png')
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

PATH = "./lenet5_net_15.pth"
BATCH_SIZE = 32
feature_map = []

def feature_map_extract(self, input):
    global feature_map
    feature_map = input[0][:10]

def data_iter(dataset_loader, device = device):
    with torch.no_grad():
        dataiter = iter(dataset_loader)
        data = dataiter.next()
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs[0].data, 1)
        wrt_str = "Network Outputs: {}\n\n Labels: {}\n\n Predicted Labels: {}\n\n".format(outputs[0][:10],labels[:10],predicted[:10]) 
        file = open("outputs.txt", "w")
        file.write(wrt_str)
        file.close()

# define transforms
# transforms.ToTensor() automatically scales the images to [0,1] range
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

net = torch.load(PATH)
net.eval()
net.to(device)

for name,module in net.named_modules():
    if( name == "feature_extractor.0" ):
        module.register_forward_pre_hook(feature_map_extract)

data_iter(valid_loader)

data_shape = feature_map.shape
shape = (120,32,32)

for i1 in range(data_shape[0]):
    val_str = "image[120][32][32] = {"
    for i2 in range(shape[0]):
        val_str += "{"
        for i3 in range(shape[1]):
            val_str += "{"
            for i4 in range(shape[2]):
                if( i2 < data_shape[1]):
                    val_str += " " + str(feature_map[i1][i2][i3][i4].item()) + ","
                else:
                    val_str += " 0,"
            val_str = val_str[:-1] + "},\n"
        val_str = val_str[:-2] + "},\n"
    val_str = val_str[:-2] + "}\n\n"
    act_file = open("Input_image" + str(i1+1) + ".txt", "w")
    act_file.write(val_str)
    act_file.close()
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from Lenet5 import LeNet5
import sys
import pickle

quant_func_file_path = "D:/University Academics/Research_Works/Scripts/Dynamic Pruning Scripts/Feature Map Prominance/Scripts/Quantization_Deployed"
sys.path.append(quant_func_file_path)

from Quant_funcs import *

device = torch.device('cpu')
print('Device: ', device)

N_CLASSES = 10
PATH = "./lenet5_net_15.pth"
BATCH_SIZE = 32
QUANT_BIT_WIDTH = 8
layer_num = 0
quantized_images = []

with open ('quantization_activation_values.txt', 'rb') as fp:
    quant_act_values = pickle.load(fp)

def first_conv_quantization_to_levels_hook(self, input):
    global quantized_images
    global layer_num
    layer_num += 1
    for i,data in enumerate(input[0], 0):
        input[0][i] = quantize_array_to_levels( input[0][i], quant_act_values[ layer_num - 1 ] )
    quantized_images = input[0];

def data_iter(dataset_loader, device = device):
    with torch.no_grad():
        dataiter = iter(dataset_loader)
        data = dataiter.next()
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs[0].data, 1)
        # wrt_str = "Network Outputs: {}\n\n Labels: {}\n\n Predicted Labels: {}\n\n".format(outputs[0][:10],labels[:10],predicted[:10]) 
        # file = open("outputs.txt", "w")
        # file.write(wrt_str)
        # file.close()

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
        module.register_forward_pre_hook(first_conv_quantization_to_levels_hook)

data_iter(valid_loader)

quant_image_size = quantized_images.size()
for i1 in range(quant_image_size[0]):
    if(i1 < 10):
        file_name = "Results/Input_image_" + str(i1 + 1) + "_quantized_to_levels.txt"
        file = open(file_name, "w")
        str_array = "ap_int<" + str(QUANT_BIT_WIDTH) + "> image_array[" + str(quant_image_size[1]) + "][" + str(quant_image_size[2]) + "][" + str(quant_image_size[3]) + "] = {"
        for i2 in range(quant_image_size[1]):
            str_array += "{"
            for i3 in range(quant_image_size[2]):
                str_array += "{"
                for i4 in range(quant_image_size[3]):
                    str_array += str(int(quantized_images[i1][i2][i3][i4].item())) + ", "
                str_array = str_array[:-2] + "},\n"
            str_array = str_array[:-2] + "},\n"
        str_array = str_array[:-2] + "}"
        file.write(str_array)
        file.close()
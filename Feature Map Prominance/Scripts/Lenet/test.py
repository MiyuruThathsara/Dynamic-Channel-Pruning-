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

with open ('quantization_activation_values.txt', 'rb') as fp:
    quant_act_values = pickle.load(fp)

with open('quantization_weight_values.txt', 'rb') as fm:
    quant_weigh_values = pickle.load(fm)

N_CLASSES = 10
PATH = "./lenet5_net_15.pth"
BATCH_SIZE = 32
QUANT_BIT_WIDTH = 2

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

params = net.state_dict()
layer_names = []
for weight_name in net.state_dict():
    layer_names.append(weight_name)
    if( weight_name != "classifier.0.weight" and weight_name != "classifier.0.bias" and weight_name != "classifier.2.weight" and weight_name != "classifier.2.bias" ):
        params[ weight_name ] = quantize_array( params[ weight_name ], quant_weigh_values[ weight_name ] )

index = 0
for parameters in net.parameters():
    parameters.data = params[ layer_names[index] ].type(torch.FloatTensor)
    index += 1
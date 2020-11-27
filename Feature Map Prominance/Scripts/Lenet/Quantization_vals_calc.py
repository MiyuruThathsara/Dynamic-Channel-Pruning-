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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

N_CLASSES = 10
PATH = "./lenet5_net_15.pth"
BATCH_SIZE = 32
QUANT_BIT_WIDTH = 8
num_of_conv_layers = 3
num_of_fc_layers = 2
quant_range_list = [0] * ( num_of_conv_layers + num_of_fc_layers )
layer_num = 0

def quant_range_hook(self, input):
    global quant_range_list
    global layer_num
    with torch.no_grad():
        layer_num += 1
        quant_range_list[ layer_num - 1 ] = max( torch.max(torch.abs(input[0])).item(), quant_range_list[ layer_num - 1 ] )

def quant_range_calc_fn(dataset_loader, device = device):
    global layer_num
    with torch.no_grad():
        for i,data in enumerate(dataset_loader,1):
            layer_num = 0
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

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

# Loading model
net = torch.load(PATH)
net.eval()
net.to(device)

# Hooking quant range calculation function to convolution layers
for name,module in net.named_modules():
    if( name == "feature_extractor.0" or name == "feature_extractor.3" or name == "feature_extractor.6" or name == "classifier.0" or name == "classifier.2"):
        module.register_forward_pre_hook(quant_range_hook)

# Calculation of max value of activation map feeding into each convolutional layer
# and calculating quantization values for the defined quantization scheme 
quant_range_calc_fn(train_loader)
quant_vals = [ quant_val_list(quant_range_list[i], QUANT_BIT_WIDTH) for i in range(num_of_conv_layers + num_of_fc_layers) ]

# Writing quantization values to a file
with open('quantization_activation_values.txt', 'wb') as fp:
    pickle.dump(quant_vals, fp)

# Quantization of weights to the defined quantization scheme and writing those to a file 
quant_dict = {}
params = net.state_dict()
for weight_name in net.state_dict():
    # if( weight_name != "classifier.0.weight" and weight_name != "classifier.0.bias" and weight_name != "classifier.2.weight" and weight_name != "classifier.2.bias" ):
    #     print("Layer name: {} max_val: {}".format( weight_name, torch.max( torch.abs( params[ weight_name ] ) ).item()) )
    quant_dict[ weight_name ] =  quant_val_list( torch.max( torch.abs( params[ weight_name ] ) ).item(), QUANT_BIT_WIDTH )

with open('quantization_weight_values.txt', 'wb') as fm:
    pickle.dump(quant_dict, fm)

# Activations
# Quantization max values of each Conv\FC Layers: [1.0, 2.0500648021698, 6.805856227874756, 22.459524154663086, 31.003461837768555]

# Weights
# Layer name: feature_extractor.0.weight max_val: 0.6834233403205872
# Layer name: feature_extractor.0.bias max_val: 0.17463120818138123
# Layer name: feature_extractor.3.weight max_val: 0.8909913301467896
# Layer name: feature_extractor.3.bias max_val: 0.19078220427036285
# Layer name: feature_extractor.6.weight max_val: 1.0573877096176147
# Layer name: feature_extractor.6.bias max_val: 0.23799854516983032
# Layer name: classifier.0.weight max_val: 0.7839701771736145
# Layer name: classifier.0.bias max_val: 0.2618212401866913
# Layer name: classifier.2.weight max_val: 1.012024164199829
# Layer name: classifier.2.bias max_val: 0.2693009078502655
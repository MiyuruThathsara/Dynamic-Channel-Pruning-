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
num_of_conv_layers = 3
num_of_train_imgs = 60000
num_of_test_imgs = 10000
running_mean_list = []
count_all_list = []
map_count_all_list = []
threshold_mask = [] 
layer_num = 0
threshold = 0.36725923556285905
mean_scale = -0.3999999999999999
data_size = BATCH_SIZE
#####################################
# for test
mask_flag = 1

with open ('quantization_activation_values.txt', 'rb') as fp:
    quant_act_values = pickle.load(fp)

with open('quantization_weight_values.txt', 'rb') as fm:
    quant_weigh_values = pickle.load(fm)

def data_iter(dataset_loader, device = device):
    with torch.no_grad():
        dataiter = iter(dataset_loader)
        data = dataiter.next()
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs[0].data, 1)
        print("Outputs : {}".format(outputs[0][:10]))
        print("Labels : {}".format(labels[:10]))
        print("Prdicted Labels : {}".format(predicted[:10]))

def first_conv_quantization_hook(self, input):
    global layer_num
    global mask_flag
    if(mask_flag == 1):
        layer_num += 1
        for i,data in enumerate(input[0], 0):
            input[0][i] = quantize_array( input[0][i], quant_act_values[ layer_num - 1 ] )

def running_mean_var_thresholding_and_masking_with_fixed_mean(self, input):
    global running_mean_list
    global threshold_mask
    global layer_num
    global mask_flag
    global count_all_list
    global map_count_all_list
    global mean
    global mean_scale
    global threshold
    global data_size
    global quant_act_values
    data_size = input[0].size()[0]
    if(mask_flag == 0):
        running_mean_list.append(input[0].mean(dim=(2,3)).sum(dim=0))
    elif(mask_flag == 1):
        layer_num += 1
        quantized_input = quantize_array(input[0], quant_act_values[layer_num - 1])
        map_count = quantized_input.size()[0] * quantized_input.size()[1]
        map_count_list = [ quantized_input.size()[1] ] * quantized_input.size()[0]
        x_mean = quantize_array( mean[ layer_num - 2 ], quant_act_values[ layer_num - 1 ] ).unsqueeze(dim = 1).unsqueeze(dim = 2)
        var = ( ( quantized_input - mean_scale * x_mean ) ** 2 ).sum(dim=(2,3))
        max_var = var.max(dim=1).values
        count = ( var <= max_var.unsqueeze(dim=1) * threshold ).sum().item()
        count_list = ( var <= max_var.unsqueeze(dim=1) * threshold ).sum(dim=1).tolist()
        var[ var <= max_var.unsqueeze(dim=1) * threshold ] = 0
        var[ var > 0 ] = 1
        mask_vec = var.type(torch.int)
        count_all_list.append(count_list)
        map_count_all_list.append(map_count_list)
        threshold_mask.append(mask_vec)
    elif(mask_flag == 2):
        layer_num += 1
        mask = threshold_mask[layer_num - 1].unsqueeze(dim = 2).unsqueeze(dim = 3)
        for i,data in enumerate(input[0], 0):
            input[0][i] = input[0][i] *  mask[i]

def running_mean_abs_thresholding_and_masking_with_fixed_mean(self, input):
    global running_mean_list
    global threshold_mask
    global layer_num
    global mask_flag
    global count_all_list
    global map_count_all_list
    global mean
    global mean_scale
    global threshold
    global data_size
    global quant_act_values
    data_size = input[0].size()[0]
    if(mask_flag == 0):
        running_mean_list.append(input[0].mean(dim=(2,3)).sum(dim=0))
    elif(mask_flag == 1):
        layer_num += 1
        quantized_input = quantize_array(input[0], quant_act_values[layer_num - 1])
        map_count = quantized_input.size()[0] * quantized_input.size()[1]
        map_count_list = [ quantized_input.size()[1] ] * quantized_input.size()[0]
        x_mean = quantize_array( mean[ layer_num - 2 ], quant_act_values[ layer_num - 1 ] ).unsqueeze(dim = 1).unsqueeze(dim = 2)
        x_abs = ( quantized_input - mean_scale * x_mean ).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * threshold ).sum().item()
        count_list = ( x_abs <= max_abs.unsqueeze(dim=1) * threshold ).sum(dim=1).tolist()
        x_abs[ x_abs <= max_abs.unsqueeze(dim=1) * threshold ] = 0
        x_abs[ x_abs > 0 ] = 1
        mask_vec = x_abs.type(torch.int)
        count_all_list.append(count_list)
        map_count_all_list.append(map_count_list)
        threshold_mask.append(mask_vec)
    elif(mask_flag == 2):
        layer_num += 1
        mask = threshold_mask[layer_num - 1].unsqueeze(dim = 2).unsqueeze(dim = 3)
        for i,data in enumerate(input[0], 0):
            input[0][i] = input[0][i] *  mask[i]

def dataset_accuracy_calc(dataset_loader, device = device):
    global mask_flag
    mask_flag = None
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(dataset_loader,1):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def static_mean_calc(dataloader, num_of_cal_imgs= num_of_train_imgs):
    with torch.no_grad():
        global running_mean_list
        global mask_flag
        mask_flag = 0
        for i,data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device, dtype= torch.float)
            outputs = net(inputs)
            if(i == 0):
                running_mean_avg = running_mean_list 
            else:
                running_mean_avg = [ running_mean_avg[j] + running_mean_list[j] for j in range(len(running_mean_list)) ]
            running_mean_list = []
        mask_flag = 1
        running_mean_avg = [ running_mean_avg[k] / num_of_cal_imgs for k in range(len(running_mean_avg)) ]
    return running_mean_avg

def static_mean_accuracy_drop_calc(testloader, num_of_conv_layers = num_of_conv_layers, train = True):
    global count_all_list
    global map_count_all_list
    global threshold_mask
    global layer_num
    global mask_flag
    global data_size
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer = [ 0 ] * num_of_conv_layers
    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            images, labels = data[0].to(device), data[1].to(device)
            count_all_list = []
            map_count_all_list = []
            threshold_mask = []
            layer_num = 0
            mask_flag = 1
            temp_forward = net(images)
            layer_num = 0
            mask_flag = 2
            for k in range(num_of_conv_layers):
                for l in range(data_size):
                    if( k == 0 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    elif(k == num_of_conv_layers - 1):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) / map_count_all_list[ k - 1 ][ l ] )
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) * ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) ) / ( map_count_all_list[ k - 1 ][ l ] * map_count_all_list[ k ][ l ] ) )
            outputs = net(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        mask_flag = None
        if(train == True):
            num_of_imgs = num_of_train_imgs
        else:
            num_of_imgs = num_of_test_imgs
    return 100 * correct / total, [ ( j / num_of_imgs ) * 100 for j in tot_flop_drop_layer ]

def avg_flop_drop_calc(percent, num_of_conv_layers = num_of_conv_layers):
    data = (["conv1_layer", (32,32,5,5,1,6,1)], ["conv2_layer", (14,14,5,5,6,16,1)], ["conv3_layer", (5,5,5,5,16,120,1)])
    tot_flop = 0
    tot_drop_flop = 0
    with torch.no_grad():
        for i in range(num_of_conv_layers):
            flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5] / data[i][1][6]
            drop_flop = percent[i] * flop / 100
            tot_flop += flop
            tot_drop_flop += drop_flop
        flop_drop_percent = tot_drop_flop / tot_flop * 100
    return flop_drop_percent

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

# # Hooking intermediate layer
# for name,module in net.named_modules():
#     if( name == "feature_extractor.0" ):
#         module.register_forward_pre_hook(first_conv_quantization_hook)
#     elif( name == "feature_extractor.3" or name == "feature_extractor.6" ):
#         module.register_forward_pre_hook(running_mean_abs_thresholding_and_masking_with_fixed_mean)

# # Calculating expected accuracy
# print('Training Dataset Accuracy Calculation!!!!!!!!!!!!!')
# exp_accuracy = dataset_accuracy_calc(train_loader)
# print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))

# print('Static Mean Calculation Started!!!!!!!!!!!!')
# # Average mean calculation of each channel
# mean = static_mean_calc(train_loader)

# Weights and biases quantization
with torch.no_grad():
    params = net.state_dict()
    layer_names = []
    for weight_name in net.state_dict():
        layer_names.append(weight_name)
        # if( weight_name != "classifier.0.weight" and weight_name != "classifier.0.bias" and weight_name != "classifier.2.weight" and weight_name != "classifier.2.bias" ):
        params[ weight_name ] = quantize_array( params[ weight_name ], quant_weigh_values[ weight_name ] )

    index = 0
    for parameters in net.parameters():
        parameters.data = params[ layer_names[index] ].type(torch.FloatTensor)
        index += 1

# print('Learned Threshold Check for Test Dataset!!!!!!!')
# baseline_test_accuracy = dataset_accuracy_calc(valid_loader)
# test_dataset_accuracy = static_mean_accuracy_drop_calc(valid_loader, train = False)
# print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy[0]))
# print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy[1]))
# print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy[1])))

for name,module in net.named_modules():
    if( name == "feature_extractor.0" or name == "feature_extractor.3" or name == "feature_extractor.6" or name == "classifier.0" or name == "classifier.2" ):
        module.register_forward_pre_hook(first_conv_quantization_hook)

data_iter(valid_loader)

# Results
# Quantization of static mean ABS method for inference phase
# Target FLOP drops = 40%
# 8 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 1.0499999999999972
# Layer by Layer FLOP Drop Percentages for Test dataset: [1.0333333333333408, 42.57499999999991, 42.333125]
# Avg FLOP Drop Percentages for Test Dataset: 38.917626096491205

# 5 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 1.0300000000000011
# Layer by Layer FLOP Drop Percentages for Test dataset: [1.904999999999977, 42.87687499999987, 42.325625]
# Avg FLOP Drop Percentages for Test Dataset: 39.06394736842101

# 4 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 5.240000000000009
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 63.931875000000005, 63.931875000000005]
# Avg FLOP Drop Percentages for Test Dataset: 58.54813815789473

# 3 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 14.329999999999998
# Layer by Layer FLOP Drop Percentages for Test dataset: [11.103333333333328, 70.3314583333333, 66.849375]
# Avg FLOP Drop Percentages for Test Dataset: 63.05298245614035

# Target FLOP drops = 20%
# 8 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.030000000000001137
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 20.808125, 20.808125]
# Avg FLOP Drop Percentages for Test Dataset: 19.055861842105262

# 5 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.06999999999999318
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 23.7875, 23.7875]
# Avg FLOP Drop Percentages for Test Dataset: 21.78434210526316

# 4 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.38000000000000966
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 30.446250000000003, 30.446250000000003]
# Avg FLOP Drop Percentages for Test Dataset: 27.882355263157898

# 3 bit quantization
# Training Dataset Accuracy Calculation!!!!!!!!!!!!!
# Expected Accuracy for Calibration Dataset: 99.8
# Static Mean Calculation Started!!!!!!!!!!!!
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 2.3200000000000074
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.20500000000000007, 40.83666666666667, 40.739999999999995]
# Avg FLOP Drop Percentages for Test Dataset: 37.351456140350884
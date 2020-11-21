########################################################################
# Import Libraries
########################################################################
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import torch.utils.data as data
import csv
import pandas as pd
from skimage import io
from VGG16 import vgg16 as vgg16 
import sys

sys.path.append('D:/University Academics/Research_Works/Scripts/data')
from Cifar10_manipulator import CIFAR10_Calibration, ToTensor, Normalize

#########################################################################
# Creation of Calibration Dataset, Variable Assignment
#########################################################################
data_root = "D:/University Academics/Research_Works/Scripts/data"
PATH = ["/cifar-10-batches-py/data_batch_1","/cifar-10-batches-py/data_batch_2","/cifar-10-batches-py/data_batch_3","/cifar-10-batches-py/data_batch_4","/cifar-10-batches-py/data_batch_5"]
for com in range(len(PATH)):
    PATH[ com ] = data_root + PATH[ com ] 
csv_filename = "D:/University Academics/Research_Works/Scripts/data/Calibration_CIFAR10.csv"
num_of_cal_imgs = 10000
num_of_conv_layers = 13
num_labels = 10
data_size = 32*32*3
label_count = [0] * num_labels
batch_size = 64

##############################################################################
# Check Whether Cuda is Available
##############################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))
##################################################################################################################
# For Calibaration Dataset, dataloader
transform_cal = transforms.Compose([ToTensor(), Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
cifar10_cal = CIFAR10_Calibration(csv_filename, transform=transform_cal)
cal_dataloader = torch.utils.data.DataLoader(cifar10_cal, batch_size = batch_size, shuffle = False, num_workers = 0)
##################################################################################################################
# For Test Dataset, dataloader
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = data_root, train = False, download = False, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 0)
##################################################################################################################

############################################################################
# Import model parameters to VGG16 Network
############################################################################
PATH = 'D:/University Academics/Research_Works/Scripts/data/VGG16 models/vgg16_net_CIFAR10_100.pth'
model = torch.load(PATH)
net = vgg16( num_classes = 10 )
net.load_state_dict(model.state_dict())
net.eval()
net.to(device)
print('Model Successfully Imported!!!!!!!!!!!!!!!!!!!!!!')

###################################################################################
# Functions to Evaluate Modified Static and Dynamic Approaches
###################################################################################
def static_mean_calc(dataloader, num_of_conv_layers= num_of_conv_layers, num_of_cal_imgs= num_of_cal_imgs):
    with torch.no_grad():
        for i,data in enumerate(dataloader, 0):
            labels, inputs = data[0].to(device), data[1].to(device, dtype= torch.float)
            running_mean = net.mean_calc_forward(inputs)
            if(i == 0):
                running_mean_avg = running_mean
            else:
                running_mean_avg = [ running_mean_avg[j] + running_mean[j] for j in range(num_of_conv_layers - 1) ]
        running_mean_avg = [ running_mean_avg[k] / num_of_cal_imgs for k in range(num_of_conv_layers - 1) ]
    return running_mean_avg

def dataset_accuracy_calc(dataloader, data_type = None):
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(dataloader,1):
            if (data_type == 'Calib'):
                labels, images = data[0].to(device = device), data[1].to(device = device, dtype = torch.float)
            else:
                images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def static_mean_accuracy_drop_calc_single_iter(testloader, data_type = 'Test', Num_Conv_Layers = num_of_conv_layers):
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer = [ 0 ] * Num_Conv_Layers

    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            if(data_type == 'Test'):
                images, labels = data[0].to(device), data[1].to(device)
            elif(data_type == 'Calib'):
                labels, images = data[0].to(device), data[1].to(device, dtype= torch.float)
            else:
                raise SyntaxError('Unknown Data Type!!!!!!')
            outputs, count, count_list, map_count, map_count_list = net.var_threshold_fixed_forward_single_iter(images)
            for k in range(Num_Conv_Layers):
                for l in range(labels.size(0)):
                    if( k == 0 ):
                        tot_flop_drop_layer[ k ] += 0
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_list[ k - 1 ][ l ] - count_list[ k - 1 ][ l ] ) / map_count_list[ k - 1 ][ l ] )
            # outputs = net.masking_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, [ ( j / num_of_cal_imgs ) * 100 for j in tot_flop_drop_layer ]


def static_mean_accuracy_drop_calc(testloader, data_type = 'Test', Num_Conv_Layers = num_of_conv_layers):
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer = [ 0 ] * Num_Conv_Layers

    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            if(data_type == 'Test'):
                images, labels = data[0].to(device), data[1].to(device)
            elif(data_type == 'Calib'):
                labels, images = data[0].to(device), data[1].to(device, dtype= torch.float)
            else:
                raise SyntaxError('Unknown Data Type!!!!!!')
            outputs, count, count_list, map_count, map_count_list = net.abs_threshold_fixed_forward(images)
            for k in range(Num_Conv_Layers):
                for l in range(labels.size(0)):
                    if( k == 0 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_list[ k ][ l ] - count_list[ k ][ l ] ) / map_count_list[ k ][ l ] )
                    elif( k == Num_Conv_Layers - 1 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_list[ k - 1 ][ l ] - count_list[ k - 1 ][ l ] ) / map_count_list[ k - 1 ][ l ] )
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( ( map_count_list[ k - 1 ][ l ] - count_list[ k - 1 ][ l ] ) * ( map_count_list[ k ][ l ] - count_list[ k ][ l ] ) ) / ( map_count_list[ k - 1 ][ l ] * map_count_list[ k ][ l ] ) )
            # outputs = net.masking_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, [ ( j / num_of_cal_imgs ) * 100 for j in tot_flop_drop_layer ]

def avg_flop_drop_calc(percent, num_of_conv_layers = num_of_conv_layers):
    data = (['layer1', (32,32,3,3,3,64)], ['layer2', (32,32,3,3,64,64)], ['layer3', (16,16,3,3,64,128)], ['layer4', (16,16,3,3,128,128)], ['layer5', (8,8,3,3,128,256)], ['layer6', (8,8,3,3,256,256)], ['layer7', (8,8,3,3,256,256)], ['layer8', (4,4,3,3,256,512)], ['layer9', (4,4,3,3,512,512)], ['layer10', (4,4,3,3,512,512)], ['layer11', (2,2,3,3,512,512)], ['layer12', (2,2,3,3,512,512)], ['layer13', (2,2,3,3,512,512)])
    tot_flop = 0
    tot_drop_flop = 0
    for i in range(num_of_conv_layers):
        flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5]
        drop_flop = percent[i] * flop / 100
        tot_flop += flop
        tot_drop_flop += drop_flop
    flop_drop_percent = tot_drop_flop / tot_flop * 100
    return flop_drop_percent

############################################################################################################
# Selecting Optimum Mean Scale Factor within 1.0 and -1.0 and setting Optimum Threshold
############################################################################################################

print('Calibration Dataset Accuracy Calculation!!!!!!')
exp_accuracy = dataset_accuracy_calc(cal_dataloader, data_type='Calib')
print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))

print('Static Mean Calculation has Started for Calibration Dataset!!!!!!!!!!!!!!!!!!!')
net.mean = static_mean_calc(cal_dataloader)

print('Threshold Learning for Calibration Dataset!!!!!!!')
info = []
accuracy_drop_list = []
mean_scale_list = []
threshold_list = []
while(True):
    state = False
    while(state == False):
        print('Initial Threshold Value: {}'.format(net.threshold))
        actual_accuracy, percent = static_mean_accuracy_drop_calc_single_iter(cal_dataloader, data_type='Calib')
        print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
        target_FLOP_drop_percent = 30
        const = 0.001
        flop_drop_percent = avg_flop_drop_calc(percent)
        target_drop_error = flop_drop_percent - target_FLOP_drop_percent
        print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
        print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
        if(abs(target_drop_error) <= 1):
            state = True
            print('Optimum Threshold Value: {}'.format(net.threshold))
            # print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
            # print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
            # print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        else:
            net.threshold = net.threshold - const * target_drop_error
            print( 'Threshold: {}'.format(net.threshold) )
            print('###########################################################################')
            state = False
    pre_mean_scale = net.mean_scale
    net.mean_scale -= 0.1
    accuracy_drop = exp_accuracy - actual_accuracy
    if(net.mean_scale < -1.0):
        min_accuracy_index = accuracy_drop_list.index(min(accuracy_drop_list))
        net.mean_scale = mean_scale_list[min_accuracy_index]
        net.threshold = threshold_list[min_accuracy_index]
        print('Info List: {}'.format(info[min_accuracy_index]))
        break
    else:
        accuracy_drop_list.append(accuracy_drop)
        info.append(['Accuracy of the network on the 10000 Calibration images: %.2f %%' % (actual_accuracy), 
                      'AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent), 
                      'Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent),
                      'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
                      'Optimum Mean Scale: {}'.format(pre_mean_scale),
                      'Optimum Threshold: {}'.format(net.threshold)])
        mean_scale_list.append(pre_mean_scale)
        threshold_list.append(net.threshold)

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(testloader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc_single_iter(testloader, data_type='Test')
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1])))


# Results
##########################################################################################
# Static Mean ABS approach single iter
##########################################################################################
# Accuracy of the network on the 10000 Calibration images: 91.73 %
# AVG FLOP drop percentage for Cal Dataset: 29.0033156779661
# Layer by Layer FLOP Drop for Cal Dataset: [0.0, 19.656875, 20.33828125, 23.266328124999998, 20.25046875, 29.407148437500002, 31.894570312499997, 30.6465625, 38.03580078125, 32.870859375, 45.7898828125, 41.234453124999995, 32.526250000000005]
# FLOP drop Error for Cal Dataset: -0.9966843220339001
# Optimum Mean Scale: -0.7
# Optimum Threshold: 0.22286533662900185
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 6.6000000000000085
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 19.8321875, 20.638906249999998, 23.4534375, 20.66625, 29.605820312499997, 32.0956640625, 30.721132812500002, 38.159902343750005, 32.91234375, 45.637324218749995, 41.21603515625, 32.41939453125]
# Avg FLOP Drop Percentages for Test Dataset: 29.15443326271187
##########################################################################################
# Static Mean SD approach single iter
##########################################################################################
# Accuracy of the network on the 10000 Calibration images: 93.31 %
# AVG FLOP drop percentage for Cal Dataset: 29.305749176082863
# Layer by Layer FLOP Drop for Cal Dataset: [0.0, 20.13109375, 21.709999999999997, 22.191640625, 20.9884375, 28.535859375, 31.235507812500003, 33.35078125, 38.287832031250005, 33.27369140625, 49.2056640625, 42.37400390625, 34.2839453125]
# FLOP drop Error for Cal Dataset: -0.6942508239171374
# Optimum Mean Scale: -0.7999999999999999
# Optimum Threshold: 0.0658363682909605
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 5.530000000000001
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 20.244375, 21.9365625, 22.3371875, 21.250703125, 28.6606640625, 31.4453515625, 33.4928125, 38.4416015625, 33.2995703125, 49.10630859375, 42.393359375, 34.25767578125]
# Avg FLOP Drop Percentages for Test Dataset: 29.433747645951026
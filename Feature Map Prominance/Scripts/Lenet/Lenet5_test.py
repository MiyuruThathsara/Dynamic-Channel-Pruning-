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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

N_CLASSES = 10
PATH = "./lenet5_net_15.pth"
BATCH_SIZE = 32
num_of_conv_layers = 3
num_of_train_imgs = 60000
num_of_test_imgs = 10000
running_mean_list = []
count_all_list = []
map_count_all_list = []
threshold_mask = []
layer_num = 0
threshold = 0.3
mean_scale = 1
data_size = BATCH_SIZE

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
    data_size = input[0].size()[0]
    if(mask_flag == 0):
        running_mean_list.append(input[0].mean(dim=(2,3)).sum(dim=0))
    elif(mask_flag == 1):
        layer_num += 1
        map_count = input[0].size()[0] * input[0].size()[1]
        map_count_list = [ input[0].size()[1] ] * input[0].size()[0]
        x_mean = mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        var = ( ( input[0] - mean_scale * x_mean ) ** 2 ).sum(dim=(2,3))
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
    data_size = input[0].size()[0]
    if(mask_flag == 0):
        running_mean_list.append(input[0].mean(dim=(2,3)).sum(dim=0))
    elif(mask_flag == 1):
        layer_num += 1
        map_count = input[0].size()[0] * input[0].size()[1]
        map_count_list = [ input[0].size()[1] ] * input[0].size()[0]
        x_mean = mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        x_abs = ( input[0] - mean_scale * x_mean ).abs().sum(dim=(2,3))
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

# Hooking intermediate layer
for name,module in net.named_modules():
    if( name == "feature_extractor.3" or name == "feature_extractor.6" ):
        module.register_forward_pre_hook(running_mean_abs_thresholding_and_masking_with_fixed_mean)

# Calculating expected accuracy
print('Training Dataset Accuracy Calculation!!!!!!!!!!!!!')
exp_accuracy = dataset_accuracy_calc(train_loader)
print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))

print('Static Mean Calculation Started!!!!!!!!!!!!')
# Average mean calculation of each channel
mean = static_mean_calc(train_loader)

print('Threshold Learning for Calibration Dataset!!!!!!!')
info = []
accuracy_drop_list = []
mean_scale_list = []
threshold_list = []
while(True):
    state = False
    while(state == False):
        actual_accuracy, percent = static_mean_accuracy_drop_calc(train_loader)
        print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
        target_FLOP_drop_percent = 40
        const = 0.001
        flop_drop_percent = avg_flop_drop_calc(percent)
        target_drop_error = flop_drop_percent - target_FLOP_drop_percent
        print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
        print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
        print('Accuracy Drop: {}'.format(exp_accuracy - actual_accuracy))
        if(abs(target_drop_error) <= 1):
            state = True
            print('Optimum Threshold Value: {}'.format(threshold))
            # print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
            # print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
            # print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        else:
            threshold = threshold - const * target_drop_error
            print( 'Threshold: {}'.format(threshold) )
            print('###########################################################################')
            state = False
    pre_mean_scale = mean_scale
    mean_scale -= 0.1
    accuracy_drop = exp_accuracy - actual_accuracy
    if(mean_scale < -1.0):
        min_accuracy_index = accuracy_drop_list.index(min(accuracy_drop_list))
        mean_scale = mean_scale_list[min_accuracy_index]
        threshold = threshold_list[min_accuracy_index]
        print('Information List: {}'.format(info[min_accuracy_index]))
        break
    else:
        accuracy_drop_list.append(accuracy_drop)
        info.append(['Accuracy of the network on the 10000 Calibration images: %.2f %%' % (actual_accuracy), 
                      'AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent), 
                      'Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent),
                      'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
                      'Optimum Mean Scale: {}'.format(pre_mean_scale),
                      'Optimum Threshold: {}'.format(threshold)])
        mean_scale_list.append(pre_mean_scale)
        threshold_list.append(threshold)

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(valid_loader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc(valid_loader, train = False)
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1])))

# Results
# Double Iteration Static Mean
# Expected accuracy: 99.80 %
# Accuracy of the network on the 10000 Calibration images: 99.71 %
# AVG FLOP drop percentage for Cal Dataset: 19.09516447368421
# Layer by Layer FLOP Drop for Cal Dataset: [0.0, 20.851041666666667, 20.851041666666667]
# FLOP drop Error for Cal Dataset: -0.9048355263157895
# Optimum Mean Scale: -0.09999999999999987
# Optimum Threshold: 0.17814546637426915
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.060000000000002274
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 20.819375, 20.819375]
# Avg FLOP Drop Percentages for Test Dataset: 19.066164473684207

# Accuracy of the network on the 10000 Calibration images: 98.85 %
# AVG FLOP drop percentage for Cal Dataset: 39.04894033260248
# Layer by Layer FLOP Drop for Cal Dataset: [1.0158333333333351, 42.71335069444498, 42.48072916666666]
# FLOP drop Error for Cal Dataset: -0.9510596673975229
# Optimum Mean Scale: -0.3999999999999999
# Optimum Threshold: 0.36725923556285905
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.9899999999999949
# Layer by Layer FLOP Drop Percentages for Test dataset: [1.066666666666675, 42.58020833333324, 42.331875000000004]
# Avg FLOP Drop Percentages for Test Dataset: 38.920953947368396

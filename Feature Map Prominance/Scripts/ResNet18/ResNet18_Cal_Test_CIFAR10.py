import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from ResNet18 import *
import sys

sys.path.append('D:/University Academics/Research_Works/Scripts/data')
from Cifar10_manipulator import CIFAR10_Calibration, ToTensor, Normalize

###################################################################
running_mean_list = []
def feature_map_running_mean(self, input):
    global running_mean_list
    running_mean_list.append(input[0].mean(dim=(2,3)).sum(dim=0))
###################################################################
###################################################################
count_all_list = []
map_count_all_list = []
threshold_mask = []
layer_num = 0
threshold = 0.10
mean_scale = 1
mask_flag = False
def activation_vol_masking(self, input):
    global threshold_mask
    global layer_num
    global mask_flag
    if(mask_flag):
        layer_num += 1
        mask = threshold_mask[layer_num - 1].unsqueeze(dim = 2).unsqueeze(dim = 3)
        input = (input[0] * mask)

def var_threshold(self, input):
    global threshold
    map_count = input[0].size()[0] * input[0].size()[1]
    var = input[0].var(dim=(2,3))
    max_var = var.max(dim=1).values
    input[0][ var <= max_var.unsqueeze(dim=1) * threshold ] = 0
    count = ( var <= max_var.unsqueeze(dim=1) * threshold ).sum().item()

def var_threshold_with_fixed_mean(self, input):
    global count_all_list
    global map_count_all_list
    global threshold_mask
    global layer_num
    global mean
    global mean_scale
    global threshold
    global mask_flag
    if(mask_flag== False):
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

def var_threshold_with_dynamic_mean(self, input):
    global threshold_mask
    global threshold
    map_count = input[0].size()[0] * input[0].size()[1]
    var = input[0].var(dim=(2,3))
    max_var = var.max(dim=1).values
    count = ( var <= max_var.unsqueeze(dim=1) * 0.03 ).sum().item()
    var[ var <= max_var.unsqueeze(dim=1) * threshold ] = 0
    var[ var > 0 ] = 1
    mask_vec = var.type(torch.int)
    threshold_mask.append(mask_vec)

def abs_threshold(self, input):
    global threshold
    map_count = input[0].size()[0] * input[0].size()[1]
    x_abs = (input[0] - input[0].mean(dim=(2,3), keepdim=True)).abs().sum(dim=(2,3))
    max_abs = x_abs.max(dim=1).values
    input[0][ x_abs <= max_abs.unsqueeze(dim=1) * threshold ] = 0
    count = ( x_abs <= max_abs.unsqueeze(dim=1) * threshold ).sum().item()

def abs_threshold_with_fixed_mean(self, input):
    global count_all_list
    global map_count_all_list
    global threshold_mask
    global layer_num
    global mean
    global mean_scale
    global threshold
    global mask_flag
    if(mask_flag==False):
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

def abs_threshold_with_dynamic_mean(self, input):
    global threshold_mask
    global threshold
    map_count = input[0].size()[0] * input[0].size()[1]
    x_abs = (input[0] - input[0].mean(dim=(2,3), keepdim=True)).abs().sum(dim=(2,3))
    max_abs = x_abs.max(dim=1).values
    count = ( x_abs <= max_abs.unsqueeze(dim=1) * threshold ).sum().item()
    x_abs[ x_abs <= max_abs.unsqueeze(dim=1) * threshold ] = 0
    x_abs[ x_abs > 0 ] = 1
    mask_vec = x_abs.type(torch.int)
    threshold_mask.append(mask_vec)

###############################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))
###############################################################################
num_of_cal_imgs = 10000
num_of_conv_layers = 17
data_root = '../../../../data'
csv_filename = data_root + '/Calibration_CIFAR10.csv'
batch_size = 4
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

PATH = './resnet18_epoch_14.pth'
net = ResNet18()
model = torch.load(PATH)
net.load_state_dict(model.state_dict())
net.eval()
net.to(device)

def static_mean_calc(dataloader, num_of_conv_layers= num_of_conv_layers, num_of_cal_imgs= num_of_cal_imgs):
    with torch.no_grad():
        global running_mean_list
        for i,data in enumerate(dataloader, 0):
            labels, inputs = data[0].to(device), data[1].to(device, dtype= torch.float)
            outputs = net(inputs)
            if(i == 0):
                running_mean_avg = running_mean_list
                running_mean_list = []
            else:
                running_mean_avg = [ running_mean_avg[j] + running_mean_list[j] for j in range(num_of_conv_layers - 1) ]
                running_mean_list = []
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

def static_mean_accuracy_drop_calc(testloader, data_type = 'Test', Num_Conv_Layers = num_of_conv_layers):
    global count_all_list
    global map_count_all_list
    global layer_num
    global mask_flag
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
            layer_num = 0
            mask_flag = False
            temp_forward = net(images)
            layer_num = 0
            mask_flag = True
            for k in range(Num_Conv_Layers):
                for l in range(batch_size):
                    if( k == 0 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    elif( k == Num_Conv_Layers - 1 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) / map_count_all_list[ k - 1 ][ l ] )
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) * ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) ) / ( map_count_all_list[ k - 1 ][ l ] * map_count_all_list[ k ][ l ] ) )
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if(i % 200 == 199):
                print('Batch size: {}'.format( i + 1 ))
    return 100 * correct / total, [ ( j / i ) * 100 for j in tot_flop_drop_layer ]

def avg_flop_drop_calc(percent, num_of_conv_layers = num_of_conv_layers):
    data = (['conv_layer', (224,224,7,7,3,64)], ['layer1_block_0_conv1', (112,112,3,3,64,64)], ['layer1_block_0_conv_2', (112,112,3,3,64,64)], ['layer1_block_1_conv_1', (112,112,3,3,64,64)], ['layer1_block_1_conv_2', (112,112,3,3,64,64)], ['layer2_block_0_conv_1', (112,112,3,3,64,128)], ['layer2_block_0_conv_2', (56,56,3,3,128,128)], ['layer2_block_1_conv_1', (56,56,3,3,128,128)], ['layer2_block_1_conv_2', (56,56,3,3,128,128)], ['layer3_block_0_conv_1', (56,56,3,3,128,256)], ['layer3_block_0_conv_2', (28,28,3,3,256,256)], ['layer3_block_1_conv_1', (28,28,3,3,256,256)], ['layer3_block_1_conv_2', (28,28,3,3,256,256)], ['layer4_block_0_conv_1', (28,28,3,3,256,512)], ['layer4_block_0_conv_2', (14,14,3,3,512,512)], ['layer4_block_1_conv_1', (14,14,3,3,512,512)], ['layer4_block_1_conv_2', (14,14,3,3,512,512)])
    tot_flop = 0
    tot_drop_flop = 0
    for i in range(num_of_conv_layers):
        flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5]
        drop_flop = percent[i] * flop / 100
        tot_flop += flop
        tot_drop_flop += drop_flop
    flop_drop_percent = tot_drop_flop / tot_flop * 100
    return flop_drop_percent
################################################################################
# Hook Feature Map Running Mean fn
################################################################################
for name, module in net.named_modules():
    if(isinstance(module, nn.Conv2d)):
        if(name != 'conv1' and name != 'layer2.0.shortcut.0' and name != 'layer3.0.shortcut.0' and name != 'layer4.0.shortcut.0'):
            module.register_forward_pre_hook(feature_map_running_mean)
################################################################################
print('Calibration Dataset Accuracy Calculation!!!!!!!!!!!!!')
exp_accuracy = dataset_accuracy_calc(cal_dataloader, data_type='Calib')
print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))
################################################################################
print('Static Mean Calculation Started!!!!!!!!!!!!')
mean = static_mean_calc(cal_dataloader)
################################################################################
# Hook Thresholding fn and Masking fn
################################################################################
for name, module in net.named_modules():
    if(isinstance(module, nn.Conv2d)):
        if(name != 'conv1' and name != 'layer2.0.shortcut.0' and name != 'layer3.0.shortcut.0' and name != 'layer4.0.shortcut.0'):
            module.register_forward_pre_hook(var_threshold_with_fixed_mean)
            module.register_forward_pre_hook(activation_vol_masking)

print('Threshold Learning for Calibration Dataset!!!!!!!')
info = []
accuracy_drop_list = []
mean_scale_list = []
threshold_list = []
while(True):
    state = False
    while(state == False):
        print('Initial Threshold Value: {}'.format(threshold))
        actual_accuracy, percent = static_mean_accuracy_drop_calc(cal_dataloader, data_type='Calib')
        print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
        target_FLOP_drop_percent = 20
        const = 0.001
        flop_drop_percent = avg_flop_drop_calc(percent)
        target_drop_error = flop_drop_percent - target_FLOP_drop_percent
        print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
        print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
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
        threshold_list.append(net.threshold)

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(testloader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc(testloader, data_type='Test')
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1])))
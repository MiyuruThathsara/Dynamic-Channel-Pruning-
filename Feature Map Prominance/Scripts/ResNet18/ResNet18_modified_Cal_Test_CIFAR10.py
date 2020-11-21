import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from ResNet18_modified import *
from pandas.core.common import flatten
import sys

sys.path.append('D:/University Academics/Research_Works/Scripts/data')
from Cifar10_manipulator import CIFAR10_Calibration, ToTensor, Normalize

###############################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))
###############################################################################
num_of_cal_imgs = 10000
num_of_conv_layers = 17
num_of_int_layers = 3
num_of_layers = 4
num_of_blocks_per_layer = 2
num_of_conv_layers_per_block = 2
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

def modified_static_mean_calc(dataloader, num_of_conv_layers= num_of_conv_layers, num_of_cal_imgs= num_of_cal_imgs):
    with torch.no_grad():
        for i,data in enumerate(dataloader, 0):
            labels, inputs = data[0].to(device), data[1].to(device, dtype= torch.float)
            running_mean = net.mean_calc_forward(inputs)
            if(i == 0):
                running_mean_avg = running_mean
            else:
                for i1 in range(num_of_layers):
                    for i2 in range(num_of_blocks_per_layer):
                        for i3 in range(num_of_conv_layers_per_block):
                            running_mean_avg[i1][i2][i3] = running_mean_avg[i1][i2][i3] * i
                            running_mean_avg[i1][i2][i3] += running_mean[i1][i2][i3]
                            running_mean_avg[i1][i2][i3] = running_mean_avg[i1][i2][i3] / (i + 1)
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

def static_mean_accuracy_drop_calc(testloader, data_type = 'Test', Num_Conv_Layers = num_of_conv_layers, Num_of_Int_Layers = num_of_int_layers):
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer = [ 0 ] * Num_Conv_Layers
    tot_flop_drop_int_layer = [ 0 ] * Num_of_Int_Layers

    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            if(data_type == 'Test'):
                images, labels = data[0].to(device), data[1].to(device)
            elif(data_type == 'Calib'):
                labels, images = data[0].to(device), data[1].to(device, dtype= torch.float)
            else:
                raise SyntaxError('Unknown Data Type!!!!!!')
            count, count_list, map_count, map_count_list = net.abs_threshold_fixed_forward(images)
            for k in range(Num_Conv_Layers):
                for l in range(batch_size):
                    if( k == 0 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_list[ k ][ l ] - count_list[ k ][ l ] ) / map_count_list[ k ][ l ] )
                    elif( k == Num_Conv_Layers - 1 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_list[ k - 1 ][ l ] - count_list[ k - 1 ][ l ] ) / map_count_list[ k - 1 ][ l ] )
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( ( map_count_list[ k - 1 ][ l ] - count_list[ k - 1 ][ l ] ) * ( map_count_list[ k ][ l ] - count_list[ k ][ l ] ) ) / ( map_count_list[ k - 1 ][ l ] * map_count_list[ k ][ l ] ) )
                    if(k == 5):
                        tot_flop_drop_int_layer[ 0 ] += 1 - ( ( map_count_list[ k ][ l ] - count_list[ k ][ l ] ) / map_count_list[ k ][ l ] )
                    elif(k == 9):
                        tot_flop_drop_int_layer[ 1 ] += 1 - ( ( map_count_list[ k ][ l ] - count_list[ k ][ l ] ) / map_count_list[ k ][ l ] )
                    elif(k == 13):
                        tot_flop_drop_int_layer[ 2 ] += 1 - ( ( map_count_list[ k ][ l ] - count_list[ k ][ l ] ) / map_count_list[ k ][ l ] )
            outputs = net.masking_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if(i % 200 == 199):
                print('Batch size: {}'.format( i + 1 ))
    return 100 * correct / total, [ ( j / num_of_cal_imgs ) * 100 for j in tot_flop_drop_layer ], [ ( l / num_of_cal_imgs ) * 100 for l in tot_flop_drop_int_layer ]

def avg_flop_drop_calc(percent, int_percent, num_of_conv_layers = num_of_conv_layers):
    data = (['conv_layer', (32,32,3,3,3,64)], ['layer1_block_0_conv1', (32,32,3,3,64,64)], ['layer1_block_0_conv_2', (32,32,3,3,64,64)], ['layer1_block_1_conv_1', (32,32,3,3,64,64)], ['layer1_block_1_conv_2', (32,32,3,3,64,64)], ['layer2_block_0_conv_1', (32,32,3,3,64,128)], ['layer2_block_0_conv_2', (16,16,3,3,128,128)], ['layer2_block_1_conv_1', (16,16,3,3,128,128)], ['layer2_block_1_conv_2', (16,16,3,3,128,128)], ['layer3_block_0_conv_1', (16,16,3,3,128,256)], ['layer3_block_0_conv_2', (8,8,3,3,256,256)], ['layer3_block_1_conv_1', (8,8,3,3,256,256)], ['layer3_block_1_conv_2', (8,8,3,3,256,256)], ['layer4_block_0_conv_1', (8,8,3,3,256,512)], ['layer4_block_0_conv_2', (4,4,3,3,512,512)], ['layer4_block_1_conv_1', (4,4,3,3,512,512)], ['layer4_block_1_conv_2', (4,4,3,3,512,512)])
    int_data = (['layer2.0.shortcut.0', (32,32,1,1,64,128)], ['layer3.0.shortcut.0', (16,16,1,1,128,256)], ['layer4.0.shortcut.0', (8,8,1,1,256,512)])
    tot_flop = 0
    tot_drop_flop = 0
    for i in range(num_of_conv_layers):
        flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5]
        drop_flop = percent[i] * flop / 100
        tot_flop += flop
        tot_drop_flop += drop_flop
    for j in range(len(int_percent)):
        flop = 2 * int_data[j][1][0] * int_data[j][1][1] * int_data[j][1][2] * int_data[j][1][3] * int_data[j][1][4] * int_data[j][1][5]
        drop_flop = int_percent[j] * flop / 100
        tot_flop += flop
        tot_drop_flop += drop_flop
    flop_drop_percent = tot_drop_flop / tot_flop * 100
    return flop_drop_percent

print('Calibration Dataset Accuracy Calculation!!!!!!')
exp_accuracy = dataset_accuracy_calc(cal_dataloader, data_type='Calib')
print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))

print('Static Mean Calculation has Started for Calibration Dataset!!!!!!!!!!!!!!!!!!!')
avg_mean = modified_static_mean_calc(cal_dataloader)
net.static_mean_feed(avg_mean)

actual_accuracy, percent, int_percent = static_mean_accuracy_drop_calc(cal_dataloader, data_type='Calib')
print('Actual Accuracy: {}'.format(actual_accuracy))
print('Layer by Layer FLOP drop percentage: {}'.format(percent))

avg_flop_drop = avg_flop_drop_calc(percent, int_percent)
print('AVG FLOP drop: {}'.format(avg_flop_drop))

print('Threshold Learning for Calibration Dataset!!!!!!!')
info = []
accuracy_drop_list = []
mean_scale_list = []
threshold_list = []
while(True):
    state = False
    while(state == False):
        actual_accuracy, percent, int_percent = static_mean_accuracy_drop_calc(cal_dataloader, data_type='Calib')
        print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
        target_FLOP_drop_percent = 20
        const = 0.001
        flop_drop_percent = avg_flop_drop_calc(percent, int_percent)
        target_drop_error = flop_drop_percent - target_FLOP_drop_percent
        print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
        print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        print('Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent))
        print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
        print('Accuracy Drop: {}'.format(exp_accuracy - actual_accuracy))
        if(abs(target_drop_error) <= 1):
            state = True
            print('Optimum Threshold Value: {}'.format(net.threshold_access()))
            # print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
            # print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
            # print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        else:
            threshold = net.threshold_access() - const * target_drop_error
            net.threshold_feed(threshold)
            print( 'Threshold: {}'.format(threshold) )
            print('###########################################################################')
            state = False
    pre_mean_scale = net.mean_scale_access()
    mean_scale = net.mean_scale_access() - 0.1
    net.mean_scale_feed(mean_scale)
    accuracy_drop = exp_accuracy - actual_accuracy
    if(mean_scale < -1.0):
        min_accuracy_index = accuracy_drop_list.index(min(accuracy_drop_list))
        net.mean_scale_feed(mean_scale_list[min_accuracy_index])
        net.threshold_feed(threshold_list[min_accuracy_index])
        print('Information List: {}'.format(info))
        break
    else:
        accuracy_drop_list.append(accuracy_drop)
        print('Mean Scale: {}'.format(pre_mean_scale))
        info.append(['Accuracy of the network on the 10000 Calibration images: %.2f %%' % (actual_accuracy), 
                      'AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent), 
                      'Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent),
                      'Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent),
                      'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
                      'Optimum Mean Scale: {}'.format(pre_mean_scale),
                      'Optimum Threshold: {}'.format(threshold)])
        mean_scale_list.append(pre_mean_scale)
        threshold_list.append(threshold)

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(testloader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc(testloader, data_type='Test')
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1],test_dataset_accuracy_drop[2])))
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Custom_Net import Custom_Net
import sys

sys.path.append('D:/University Academics/Research_Works/Scripts/data')
from Cifar10_manipulator import CIFAR10_Calibration, ToTensor, Normalize

num_of_cal_imgs = 10000
num_of_conv_layers = 4
data_root = '../../../../data'
csv_filename = data_root + '/Calibration_CIFAR10.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device which runs the Model CustomNet: {}'.format(device))
##################################################################################################################
# For Calibaration Dataset, dataloader
transform_cal = transforms.Compose([ToTensor(), Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
cifar10_cal = CIFAR10_Calibration(csv_filename, transform=transform_cal)
cal_dataloader = torch.utils.data.DataLoader(cifar10_cal, batch_size = 1, shuffle = False, num_workers = 0)
##################################################################################################################
# For Test Dataset, dataloader
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = data_root, train = False, download = False, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 0)
##################################################################################################################

PATH = './custom_net_100_epochs.pth'
net = Custom_Net()
model = torch.load(PATH)
net.load_state_dict(model.state_dict())
net.eval()
net.to(device)

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

def static_mean_accuracy_drop_calc(testloader, data_type = 'Test'):
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer1 = 0
    tot_flop_drop_layer2 = 0
    tot_flop_drop_layer3 = 0
    tot_flop_drop_layer4 = 0

    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            if(data_type == 'Test'):
                images, labels = data[0].to(device), data[1].to(device)
            elif(data_type == 'Calib'):
                labels, images = data[0].to(device), data[1].to(device, dtype= torch.float)
            else:
                raise SyntaxError('Unknown Data Type!!!!!!')
            count1, map_count1, count2, map_count2, count3, map_count3 = net.abs_threshold_forward(images)
            tot_flop_drop_layer1 += 1 - ( ( map_count1 - count1 ) / map_count1 )
            tot_flop_drop_layer2 += 1 - ( ( map_count1 - count1 ) * ( map_count2 - count2 ) ) / ( map_count1 * map_count2 )
            tot_flop_drop_layer3 += 1 - ( ( map_count2 - count2 ) * ( map_count3 - count3 ) ) / ( map_count2 * map_count3 )
            tot_flop_drop_layer4 += 1 - ( ( map_count3 - count3 ) / map_count3 )
            outputs = net.masking_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        flop_drop_percent_layer1 = tot_flop_drop_layer1 / i * 100
        flop_drop_percent_layer2 = tot_flop_drop_layer2 / i * 100
        flop_drop_percent_layer3 = tot_flop_drop_layer3 / i * 100
        flop_drop_percent_layer4 = tot_flop_drop_layer4 / i * 100
    return 100 * correct / total, [ flop_drop_percent_layer1, flop_drop_percent_layer2, flop_drop_percent_layer3, flop_drop_percent_layer4 ]

def avg_flop_drop_calc(percent, num_of_conv_layers = num_of_conv_layers):
    data = (['layer1', (32,32,3,3,3,6)], ['layer2', (16,16,3,3,6,16)], ['layer3', (8,8,5,5,16,32)], ['layer4', (8,8,3,3,32,32)])
    tot_flop = 0
    tot_drop_flop = 0
    for i in range(num_of_conv_layers):
        flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5]
        drop_flop = percent[i] * flop / 100
        tot_flop += flop
        tot_drop_flop += drop_flop
    flop_drop_percent = tot_drop_flop / tot_flop * 100
    return flop_drop_percent

print('Calibration Dataset Accuracy Calculation!!!!!!')
exp_accuracy = dataset_accuracy_calc(cal_dataloader, data_type='Calib')
print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))

print('Static Mean Calculation has Started for Calibration Dataset!!!!!!!!!!!!!!!!!!!')
net.mean = static_mean_calc(cal_dataloader)

print('Threshold Learning for Calibration Dataset!!!!!!!')
mean_scale_flag = 0
pos_acc_drop = 0
neg_acc_drop = 0
first_run_flag = 0
while(True):
    if(mean_scale_flag == 0):
        net.mean_scale = 1.1
    elif(mean_scale_flag == 1):
        net.mean_scale = 0.9
    else:
        if(first_run_flag == 0):
            if(pos_acc_drop <= neg_acc_drop):
                pre_net_mean_scale = 1.1
                net.mean_scale = 1.2
                pre_accuracy_drop = pos_acc_drop
                pre_net_threshold = pos_threshold
            else:
                pre_net_mean_scale = 0.9
                net.mean_scale = 0.8
                pre_accuracy_drop = neg_acc_drop
                pre_net_threshold = neg_threshold
            first_run_flag = 1
        else:
            if(pos_acc_drop <= neg_acc_drop):
                net.mean_scale += 0.1
            else:
                net.mean_scale -= 0.1
    state = False
    while(state == False):
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
            print('Optimum Threshold Value: {}'.format(net.threshold))
            # print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
            # print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
            # print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        else:
            net.threshold = net.threshold - const * target_drop_error
            print( 'Threshold: {}'.format(net.threshold) )
            print('###########################################################################')
            state = False

    if(mean_scale_flag == 0):
        pos_acc_drop = exp_accuracy - actual_accuracy
        pos_threshold = net.threshold
        mean_scale_flag = 1
    elif(mean_scale_flag == 1):
        neg_acc_drop = exp_accuracy - actual_accuracy
        neg_threshold = net.threshold
        mean_scale_flag = 2
    else: 
        accuracy_drop = exp_accuracy - actual_accuracy
        if(pre_accuracy_drop < accuracy_drop):
            net.mean_scale = pre_net_mean_scale
            net.threshold = pre_net_threshold
            print('Optimum Solution: {}'.format(pre_info_list))
            break
        else:
            pre_net_mean_scale = net.mean_scale
            pre_net_threshold = net.threshold
            pre_info_list = ['Accuracy of the network on the 10000 Calibration images: %.2f %%' % (actual_accuracy), 
                            'AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent), 
                            'Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent),
                            'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
                            'Optimum Mean Scale: {}'.format(net.mean_scale)]
            pre_accuracy_drop = exp_accuracy - actual_accuracy

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(testloader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc(testloader, data_type='Test')
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1])))

# Static ABS Method
# Calibration Dataset Accuracy Calculation!!!!!!
# Expected Accuracy for Calibration Dataset: 99.73
# Static Mean Calculation has Started for Calibration Dataset!!!!!!!!!!!!!!!!!!!
# Threshold Learning for Calibration Dataset!!!!!!!

# Static L1 Norm Method with mean_scale = 0.0
# Accuracy of the network on the 10000 Calibration images: 95.41 %
# AVG FLOP drop percentage for Cal Dataset: 20.946240022805164
# Layer by Layer FLOP Drop for Cal Dataset: [18.8883333333352, 26.8717708333331, 23.746015625, 15.414375]
# FLOP drop Error for Cal Dataset: 0.9462400228051635
# Optimum Threshold Value: 0.16652551774512395
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.75
# Layer by Layer FLOP Drop Percentages for Test dataset: [18.963333333335207, 27.06177083333312, 24.0312109375, 15.5703125]
# Avg FLOP Drop Percentages for Test Dataset: 21.157851339794902

# Static ABS Method with mean scale
# Accuracy of the network on the 10000 Calibration images: 96.10 %
# AVG FLOP drop percentage for Cal Dataset: 20.857326824401543
# Layer by Layer FLOP Drop for Cal Dataset: [18.060000000002148, 25.581562499999826, 23.7638671875, 15.835625]
# FLOP drop Error for Cal Dataset: 0.8573268244015431
# Optimum Threshold Value: 0.24906243051590288
# Optimum Mean Scale: 0.40000000000000013
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 1.1099999999999994
# Layer by Layer FLOP Drop Percentages for Test dataset: [18.21333333333547, 25.733437499999884, 23.9851953125, 15.9146875]
# Avg FLOP Drop Percentages for Test Dataset: 21.01710305017122

# Static STD method with mean scale
# Accuracy of the network on the 10000 Calibration images: 98.11 %
# AVG FLOP drop percentage for Cal Dataset: 19.069852480045824 
# Layer by Layer FLOP Drop for Cal Dataset: [16.803333333335317, 22.536562500000233, 21.2029296875, 15.4446875]
# FLOP drop Error for Cal Dataset: -0.9301475199541756
# Optimum Mean Scale: -0.8999999999999998
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.28999999999999204
# Layer by Layer FLOP Drop Percentages for Test dataset: [16.875000000002014, 22.864479166666786, 21.920976562499998, 15.9234375]
# Avg FLOP Drop Percentages for Test Dataset: 19.601572833523576

# Static ABS Method
# Accuracy of the network on the 10000 Calibration images: 92 %
# AVG FLOP drop percentage for Cal Dataset: 19.99203107183601
# Layer by Layer FLOP Drop for Cal Dataset: [18.285000000002153, 24.661458333333385, 22.383671875, 15.399375000000001]
# FLOP drop Error for Cal Dataset: -0.007968928163990086
# Optimum Threshold Value: 0.3745827594070569
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 3.3400000000000034
# Layer by Layer FLOP Drop Percentages for Test dataset: [18.38000000000215, 24.849374999999924, 22.55130859375, 15.3571875]
# Avg FLOP Drop Percentages for Test Dataset: 20.08655181014842

#Static STD Method
# Accuracy of the network on the 10000 Calibration images: 94 %
# AVG FLOP drop percentage for Cal Dataset: 19.99141854332971
# Layer by Layer FLOP Drop for Cal Dataset: [17.36166666666859, 24.63468750000001, 22.717832031249998, 15.203125]
# FLOP drop Error for Cal Dataset: -0.008581456670288645
# Optimum Threshold Value: 0.11694173923887714
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 2.1499999999999915
# Layer by Layer FLOP Drop Percentages for Test dataset: [17.351666666668564, 24.643854166666536, 22.753691406250002, 15.150312499999998]
# Avg FLOP Drop Percentages for Test Dataset: 19.99063604618032
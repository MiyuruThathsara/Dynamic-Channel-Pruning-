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
            # print('Accuracy = ', (predicted == labels).sum().item()/labels.size(0))
            # if(i%200 == 0):
            #     print('Mini-Batch Count: %4d' % i)
        # print('Finished Testing!!!!!')
    # print('Accuracy of the network on the 10000 Calibration images: %.2f %%' % (100 * correct / total))
    return 100 * correct / total

def dynamic_mean_accuracy_drop_calc(testloader):
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer1 = 0
    tot_flop_drop_layer2 = 0
    tot_flop_drop_layer3 = 0
    tot_flop_drop_layer4 = 0

    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            images, labels = data[0].to(device), data[1].to(device)
            outputs, count1, map_count1, count2, map_count2, count3, map_count3 = net.dynamic_mean_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            tot_flop_drop_layer1 += 1 - ( ( map_count1 - count1 ) / map_count1 )
            tot_flop_drop_layer2 += 1 - ( ( map_count1 - count1 ) * ( map_count2 - count2 ) ) / ( map_count1 * map_count2 )
            tot_flop_drop_layer3 += 1 - ( ( map_count2 - count2 ) * ( map_count3 - count3 ) ) / ( map_count2 * map_count3 )
            tot_flop_drop_layer4 += 1 - ( ( map_count3 - count3 ) / map_count3 )
            correct += (predicted == labels).sum().item()
        #     if(i%200 == 0):
        #         print('Mini-Batch Count: %4d' % i)
        # print('Finished Testing!!!!!!')
        # print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)
        flop_drop_percent_layer1 = tot_flop_drop_layer1 / i * 100
        flop_drop_percent_layer2 = tot_flop_drop_layer2 / i * 100
        flop_drop_percent_layer3 = tot_flop_drop_layer3 / i * 100
        flop_drop_percent_layer4 = tot_flop_drop_layer4 / i * 100
    #     print('Layer1 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer1 ) )
    #     print('Layer2 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer2 ) )
    #     print('Layer3 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer3 ) )
    #     print('Layer4 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer4 ) )

    # print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    # print('FLOP Drop Percentages Layer by Layer: {}'.format([ tot_flop_drop_layer1, tot_flop_drop_layer2, tot_flop_drop_layer3, tot_flop_drop_layer4 ]))
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

print('Threshold Learning for Calibration Dataset!!!!!!!')
state = False
while (state == False):
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer1 = 0
    tot_flop_drop_layer2 = 0
    tot_flop_drop_layer3 = 0
    tot_flop_drop_layer4 = 0

    with torch.no_grad():
        for i, data in enumerate(cal_dataloader, 1):
            labels, images = data[0].to(device = device), data[1].to(device = device, dtype = torch.float)
            outputs, count1, map_count1, count2, map_count2, count3, map_count3 = net.dynamic_mean_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            tot_flop_drop_layer1 += 1 - ( ( map_count1 - count1 ) / map_count1 )
            tot_flop_drop_layer2 += 1 - ( ( map_count1 - count1 ) * ( map_count2 - count2 ) ) / ( map_count1 * map_count2 )
            tot_flop_drop_layer3 += 1 - ( ( map_count2 - count2 ) * ( map_count3 - count3 ) ) / ( map_count2 * map_count3 )
            tot_flop_drop_layer4 += 1 - ( ( map_count3 - count3 ) / map_count3 )
            correct += (predicted == labels).sum().item()
        #     if(i%200 == 0):
        #         print('Mini-Batch Count: %4d' % i)
        # print('Finished Testing!!!!!!')
        # print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)
        flop_drop_percent_layer1 = tot_flop_drop_layer1 / i * 100
        flop_drop_percent_layer2 = tot_flop_drop_layer2 / i * 100
        flop_drop_percent_layer3 = tot_flop_drop_layer3 / i * 100
        flop_drop_percent_layer4 = tot_flop_drop_layer4 / i * 100
        # print('Layer1 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer1 ) )
        # print('Layer2 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer2 ) )
        # print('Layer3 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer3 ) )
        # print('Layer4 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer4 ) )

    print('Accuracy of the network on the 10000 Calibration images: %.2f %%' % (100 * correct / total))
    target_FLOP_drop_percent = 20
    const = 0.001
    percent = [ flop_drop_percent_layer1, flop_drop_percent_layer2, flop_drop_percent_layer3, flop_drop_percent_layer4 ]
    flop_drop_percent = avg_flop_drop_calc(percent)
    target_drop_error = flop_drop_percent - target_FLOP_drop_percent
    print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
    print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
    print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
    if(abs(target_drop_error) <= 0.01):
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

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(testloader)
test_dataset_accuracy_drop = dynamic_mean_accuracy_drop_calc(testloader)
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1])))

#Expected Accuracy for Calibration Dataset: 99.73

# ABS Threshold
# Accuracy of the network on the 10000 Calibration images: 96.54 %
# Layer by Layer FLOP Drop for Cal Dataset: [18.245000000001543, 25.257083333333213, 22.3935546875, 15.17625]
# FLOP drop Error for Cal Dataset: -0.007078819840238282
# Optimum Threshold Value: 0.2828734535347724
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 0.8299999999999983
# Layer by Layer FLOP Drop Percentages for Test dataset: [18.126666666668115, 25.14156250000003, 22.43755859375, 15.200625]
# Avg FLOP Drop Percentages for Test Dataset: 19.995840578677445

# STD Threshold
# Accuracy of the network on the 10000 Calibration images: 95.61 %
# AVG FLOP drop percentage for Cal Dataset: 19.99353798460673
# Layer by Layer FLOP Drop for Cal Dataset: [18.79166666666821, 26.130937499999803, 22.39431640625, 14.695625000000001]
# FLOP drop Error for Cal Dataset: -0.006462015393271514
# Optimum Threshold Value: 0.09194537307582362
# Learned Threshold Check for Test Dataset!!!!!!!
# Test Accuracy Drop: 1.1499999999999915
# Layer by Layer FLOP Drop Percentages for Test dataset: [18.70000000000145, 26.095624999999835, 22.44423828125, 14.67]
# Avg FLOP Drop Percentages for Test Dataset: 19.995077323261235
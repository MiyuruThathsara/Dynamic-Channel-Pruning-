import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Custom_Net import Custom_Net

num_of_train_imgs = 50000
num_of_conv_layers = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root = './gdrive/My Drive/Research Project/Dynamic Channel Pruning Research/Data', train = True, download = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 0)
testset = torchvision.datasets.CIFAR10(root = './gdrive/My Drive/Research Project/Dynamic Channel Pruning Research/Data', train = False, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 0)

PATH = './gdrive/My Drive/Research Project/Dynamic Channel Pruning Research/Work/Custom Net/custom_net_30_epochs.pth'

net = Custom_Net()
model = torch.load(PATH)
net.load_state_dict(model.state_dict())
net.eval()
net.to(device)

print('Mean Calculation has Started!!!!!!!!!!!!!!!!!!!')
with torch.no_grad():
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        running_mean = net.mean_calc_forward(inputs)
        if(i == 0):
            running_mean_avg = running_mean
        else:
            running_mean_avg = [ running_mean_avg[j] + running_mean[j] for j in range(num_of_conv_layers - 1) ]
    running_mean_avg = [ running_mean_avg[k] / num_of_train_imgs for k in range(num_of_conv_layers - 1) ]

    net.mean = running_mean_avg
    # mean = sum(net.layer_mean) / len(net.layer_mean) 
    # variance = sum([((x - mean) ** 2) for x in net.layer_mean]) / len(net.layer_mean) 
    # std = variance ** 0.5
    # plt.figure()
    # plt.hist(net.layer_mean, bins = 100)
    # plt.text(1.3, 1000, 'std_deviation = %.3f, mean = %.3f' %(std,mean), style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}, fontsize = 8)
    # plt.show()
    # plt.savefig('./gdrive/My Drive/Research Project/Dynamic Channel Pruning Research/Work/Custom Net/Layer2_13.png')

print('Testing Started!!!!!!!!!!!!!!')
prev_flop_drop_percent = 0.0
state = False

while(state == False):
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer1 = 0
    tot_flop_drop_layer2 = 0
    tot_flop_drop_layer3 = 0
    tot_flop_drop_layer4 = 0

    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            images, labels = data[0].to(device), data[1].to(device)
            count1, map_count1, count2, map_count2, count3, map_count3 = net.abs_threshold_forward(images)
            tot_flop_drop_layer1 += 1 - ( ( map_count1 - count1 ) / map_count1 )
            tot_flop_drop_layer2 += 1 - ( ( map_count1 - count1 ) * ( map_count2 - count2 ) ) / ( map_count1 * map_count2 )
            tot_flop_drop_layer3 += 1 - ( ( map_count2 - count2 ) * ( map_count3 - count3 ) ) / ( map_count2 * map_count3 )
            tot_flop_drop_layer4 += 1 - ( ( map_count3 - count3 ) / map_count3 )
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if(i%200 == 0):
                print('Mini-Batch Count: %4d' % i)
        print('Finished Testing!!!!!!')
        # print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)
        flop_drop_percent_layer1 = tot_flop_drop_layer1 / i * 100
        flop_drop_percent_layer2 = tot_flop_drop_layer2 / i * 100
        flop_drop_percent_layer3 = tot_flop_drop_layer3 / i * 100
        flop_drop_percent_layer4 = tot_flop_drop_layer4 / i * 100
        print('Layer1 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer1 ) )
        print('Layer2 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer2 ) )
        print('Layer3 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer3 ) )
        print('Layer4 Total FLOP drop Percentage : %.2f %%' % ( flop_drop_percent_layer4 ) )

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('Threshold Value: ', net.threshold)

    exp_accuracy = 68
    accuracy = 100 * correct / total

    if( abs( exp_accuracy - accuracy ) < 1 ):
        data = (['layer1', (32,32,3,3,3,6)], ['layer2', (16,16,3,3,6,16)], ['layer3', (8,8,5,5,16,32)], ['layer4', (8,8,3,3,32,32)])
        percent = [ flop_drop_percent_layer1, flop_drop_percent_layer2, flop_drop_percent_layer3, flop_drop_percent_layer4 ]
        tot_flop = 0
        tot_drop_flop = 0
        for i in range(num_of_conv_layers):
            flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5]
            drop_flop = percent[i] * flop / 100
            tot_flop += flop
            tot_drop_flop += drop_flop
        flop_drop_percent = tot_drop_flop / tot_flop * 100
        print("FLOP Drop = ", flop_drop_percent)
        if( ( flop_drop_percent - prev_flop_drop_percent ) > 0 ):
            prev_flop_drop_percent = flop_drop_percent
            net.mean_scale -= 0.05
            state = False
        else:
            state = True
    else:
        net.threshold -= 0.01
        state = False


# correct = 0.0
# total = 0.0
# tot_flop_drop_layer1 = 0
# tot_flop_drop_layer2 = 0
# tot_flop_drop_layer3 = 0
# tot_flop_drop_layer4 = 0

# with torch.no_grad():
#     for i, data in enumerate(testloader, 1):
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs, count1, map_count1, count2, map_count2, count3, map_count3 = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         tot_flop_drop_layer1 += 1 - ( ( map_count1 - count1 ) / map_count1 )
#         tot_flop_drop_layer2 += 1 - ( ( map_count1 - count1 ) * ( map_count2 - count2 ) ) / ( map_count1 * map_count2 )
#         tot_flop_drop_layer3 += 1 - ( ( map_count2 - count2 ) * ( map_count3 - count3 ) ) / ( map_count2 * map_count3 )
#         tot_flop_drop_layer4 += 1 - ( ( map_count3 - count3 ) / map_count3 )
#         correct += (predicted == labels).sum().item()
#         if(i%200 == 0):
#             print('Mini-Batch Count: %4d' % i)
#     print('Finished Testing!!!!!!')
#     # print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)
#     print('Layer1 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer1 / i * 100 ) )
#     print('Layer2 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer2 / i * 100 ) )
#     print('Layer3 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer3 / i * 100 ) )
#     print('Layer4 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer4 / i * 100 ) )

# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
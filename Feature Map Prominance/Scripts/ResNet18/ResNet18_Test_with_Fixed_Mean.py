import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from ResNet18 import *

num_of_training_imgs = 50000  # Number of Training Images in dataset
num_of_layers = 13      # Number of Layers which have removed feature maps
layers_count = [ 0 ] * num_of_layers     # List which includes total removed feature maps(Out of taken into consideration) in each layer
layers_map_count = [ 0 ] * num_of_layers # List which includes total number of feature maps(Considered) in each layer
tot_flop_drop_layer = [ 0 ] * ( num_of_layers + 1 )
threshold_mask = []
tot_running_mean = []
layer_num = 0           # Number of the layer
tot_count = 0
tot_map_count = 0

########################################
running_mean_avg = []
flag = 0    # This is to keep activate and deactivate activation_threshold Masking

def forward_hook_pre( self, input ):
    global layers_count
    global layer_num
    global tot_count
    global tot_map_count
    count = 0
    map_count = 0
    max1 = torch.max(input[0].abs())
    for j,im_1 in enumerate(input[0],0):
        im = ( max1 / torch.max(im_1.abs())) * im_1
        im_size = im.size()[1] * im.size()[2]
        var = im.var(dim=(1,2))
        map_count += im.size()[0]
        max_var = max(var)
        input[0][j][ var <= (0.11 * max_var) ] = 0
        count += var[ var <= 0.11 * max_var ].size(0)
    layers_count[ layer_num ] = count
    layers_map_count[ layer_num ] = map_count
    tot_count += count
    tot_map_count += map_count
    if( layer_num % num_of_layers == ( num_of_layers - 1 )):
        layer_num = 0
    else:
        layer_num += 1

def forward_hook_pre_abs( self, input ):
    global layers_count
    global layer_num
    global tot_count
    global tot_map_count
    global num_of_layers
    count = 0
    map_count = 0
    max1 = torch.max(input[0].abs())
    for j,im_1 in enumerate(input[0],0):
        im = ( max1 / torch.max(im_1.abs())) * im_1
        im_size = im.size()[1] * im.size()[2]
        im_mean = im.mean(dim=(1,2), keepdim=True)
        im_abs = (im - im_mean).abs().sum(dim=(1,2))
        map_count += im.size()[0]
        max_abs = max(im_abs)
        input[0][j][ im_abs <= (0.33 * max_abs) ] = 0
        count += im_abs[ im_abs <= 0.33 * max_abs ].size(0)
    layers_count[ layer_num ] = count
    layers_map_count[ layer_num ] = map_count
    tot_count += count
    tot_map_count += map_count
    if( layer_num % num_of_layers == ( num_of_layers - 1 )):
        layer_num = 0
    else:
        layer_num += 1

def abs_threshold_with_fixed_mean( self, input ):
    global threshold_mask
    global layers_count
    global layer_num
    global tot_count
    global tot_map_count
    global num_of_layers
    global flag
    count = 0
    map_count = 0
    max1 = torch.max(input[0].abs())
    if(flag == 0):
        if( layer_num == 0 ):
            threshold_mask = []

        for j,im_1 in enumerate(input[0],0):
            im = ( max1 / torch.max(im_1.abs())) * im_1
            im_size = im.size()[1] * im.size()[2]
            im_mean = running_mean_avg[ layer_num ].unsqueeze(dim = 1).unsqueeze(dim = 1)
            im_abs = (im - im_mean).abs().sum(dim=(1,2))
            map_count += im.size()[0]
            max_abs = max(im_abs)
            count += im_abs[ im_abs <= 0.33 * max_abs ].size(0)
            im_abs[ im_abs <= 0.28 * max_abs ] = 0
            im_abs[ im_abs > 0 ] = 1
            mask_vec = im_abs.type(torch.int)
        layers_count[ layer_num ] = count
        layers_map_count[ layer_num ] = map_count
        tot_count += count
        tot_map_count += map_count
        threshold_mask.append(mask_vec)
        if( layer_num % num_of_layers == ( num_of_layers - 1 )):
            layer_num = 0
            flag = 1
        else:
            layer_num += 1

def feature_map_running_mean(self, input):
    global layer_num
    global num_of_layers
    global tot_running_mean

    running_mean = torch.zeros(input[0].size(1)).cuda()
    if(layer_num == 0):
        tot_running_mean = []

    for i,im in enumerate(input[0],1):
        running_mean += im.mean(dim=(1,2))
    tot_running_mean.append(running_mean)
    if( layer_num % num_of_layers == ( num_of_layers - 1 )):
        layer_num = 0
    else:
        layer_num += 1

def activation_vol_masking(self, input):
    global layer_num
    global num_of_layers
    global flag
    if(flag == 1):
        mask = threshold_mask[ layer_num ].unsqueeze(dim = 1).unsqueeze(dim = 1)
        input[0] = input[0] * mask
        if( layer_num % num_of_layers == ( num_of_layers - 1 )):
            layer_num = 0
            flag = 0
        else:
            layer_num += 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='../../../../data', train=True, transform= transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 0)
testset = torchvision.datasets.CIFAR10(root = '../../../../data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 0)

PATH = './resnet18_epoch_14.pth'
net = torch.load(PATH)
net.eval()
net.to(device)

layers = list(net._modules.items())
# print(layers)

layers[2][1][0].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[2][1][0].conv2.register_forward_pre_hook(feature_map_running_mean)
layers[2][1][1].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[2][1][1].conv2.register_forward_pre_hook(feature_map_running_mean)
# layers[3][1][0].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[3][1][0].conv2.register_forward_pre_hook(feature_map_running_mean)
layers[3][1][1].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[3][1][1].conv2.register_forward_pre_hook(feature_map_running_mean)
# layers[4][1][0].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[4][1][0].conv2.register_forward_pre_hook(feature_map_running_mean)
layers[4][1][1].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[4][1][1].conv2.register_forward_pre_hook(feature_map_running_mean)
# layers[5][1][0].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[5][1][0].conv2.register_forward_pre_hook(feature_map_running_mean)
layers[5][1][1].conv1.register_forward_pre_hook(feature_map_running_mean)
layers[5][1][1].conv2.register_forward_pre_hook(feature_map_running_mean)

with torch.no_grad():
    print("Running Mean Calculation has Started!!!!!!!!!!!!!!!!!!!")
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        if(i == 0):
            running_mean = tot_running_mean
        else:
            running_mean = [ tot_running_mean[j] + running_mean[j] for j in range(num_of_layers) ]
    running_mean_avg = [ running_mean[k] / num_of_training_imgs for k in range(num_of_layers) ]

layers[2][1][0].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[2][1][0].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[2][1][1].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[2][1][1].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)
# layers[3][1][0].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[3][1][0].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[3][1][1].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[3][1][1].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)
# layers[4][1][0].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[4][1][0].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[4][1][1].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[4][1][1].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)
# layers[5][1][0].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[5][1][0].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[5][1][1].conv1.register_forward_pre_hook(abs_threshold_with_fixed_mean)
layers[5][1][1].conv2.register_forward_pre_hook(abs_threshold_with_fixed_mean)

layers[2][1][0].conv1.register_forward_pre_hook(activation_vol_masking)
layers[2][1][0].conv2.register_forward_pre_hook(activation_vol_masking)
layers[2][1][1].conv1.register_forward_pre_hook(activation_vol_masking)
layers[2][1][1].conv2.register_forward_pre_hook(activation_vol_masking)
# layers[3][1][0].conv1.register_forward_pre_hook(activation_vol_masking)
layers[3][1][0].conv2.register_forward_pre_hook(activation_vol_masking)
layers[3][1][1].conv1.register_forward_pre_hook(activation_vol_masking)
layers[3][1][1].conv2.register_forward_pre_hook(activation_vol_masking)
# layers[4][1][0].conv1.register_forward_pre_hook(activation_vol_masking)
layers[4][1][0].conv2.register_forward_pre_hook(activation_vol_masking)
layers[4][1][1].conv1.register_forward_pre_hook(activation_vol_masking)
layers[4][1][1].conv2.register_forward_pre_hook(activation_vol_masking)
# layers[5][1][0].conv1.register_forward_pre_hook(activation_vol_masking)
layers[5][1][0].conv2.register_forward_pre_hook(activation_vol_masking)
layers[5][1][1].conv1.register_forward_pre_hook(activation_vol_masking)
layers[5][1][1].conv2.register_forward_pre_hook(activation_vol_masking)

correct = 0.0
total = 0.0
tot_map_drop = 0

print("Testing Has Started!!!!!!!")

with torch.no_grad():
    for i, data in enumerate(testloader, 1):
        tot_count = 0
        images, labels = data[0].to(device), data[1].to(device)
        for k in range(2):
            outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        for k in range(num_of_layers + 1):
          if( k == 0 ):
              tot_flop_drop_layer[ k ] += 1 - ( ( layers_map_count[ k ] - layers_count[ k ] ) / layers_map_count[ k ] )
          elif( k == num_of_layers ):
              tot_flop_drop_layer[ k ] += 1 - ( ( layers_map_count[ k - 1 ] - layers_count[ k - 1 ] ) / layers_map_count[ k - 1 ] )
          else:
              tot_flop_drop_layer[ k ] += 1 - ( ( ( layers_map_count[ k - 1 ] - layers_count[ k - 1 ] ) * ( layers_map_count[ k ] - layers_count[ k ] ) ) / ( layers_map_count[ k - 1 ] * layers_map_count[ k ] ) )
        correct += (predicted == labels).sum().item()
        # print('Count = ', tot_count, 'Map Count = ', tot_map_count, 'Accuracy = ', (predicted == labels).sum().item()/labels.size(0) * 100)
        # print(((predicted == labels).sum().item())/labels.size(0))
        if(i%200 == 0):
            print('Mini-Batch Count: %4d' % i)
    print('Finished Testing!!!!!!')
    print('Total FLOP Drop Percentage as an Average:= ', [ ( j / i ) * 100 for j in tot_flop_drop_layer ])

print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * float(correct) / total))
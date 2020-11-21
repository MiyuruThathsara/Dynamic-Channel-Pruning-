import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from ResNet18 import *

num_of_layers = 16      # Number of Layers which have removed feature maps
layers_count = [ 0 ] * num_of_layers     # List which includes total removed feature maps(Out of taken into consideration) in each layer
layers_map_count = [ 0 ] * num_of_layers # List which includes total number of feature maps(Considered) in each layer
tot_flop_drop_layer = [ 0 ] * ( num_of_layers + 1 )
layer_num = 0           # Number of the layer
tot_count = 0
tot_map_count = 0


# For STD Feature Map Removal Methodology
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
        input[0][j][ var <= (0.035 * max_var) ] = 0
        count += var[ var <= 0.035 * max_var ].size(0)
    layers_count[ layer_num ] = count
    layers_map_count[ layer_num ] = map_count
    tot_count += count
    tot_map_count += map_count
    if( layer_num % num_of_layers == 15):
        layer_num = 0
    else:
        layer_num += 1

# For ABS Feature Map Removal Methodology
def forward_hook_pre_abs( self, input ):
    global layers_count
    global layer_num
    global tot_count
    global tot_map_count
    count = 0
    map_count = 0
    max1 = torch.max(input[0].abs())
    for j,im_1 in enumerate(x,0):
        im = ( max1 / torch.max(im_1.abs())) * im_1
        im_size = im.size()[1] * im.size()[2]
        im_mean = im.mean(dim=(1,2), keepdim=True)
        im_abs = (im - im_mean).abs().sum(dim=(1,2))
        map_count += im.size()[0]
        max_abs = max(im_abs)
        input[0][j][ im_abs <= (0.15 * max_abs) ] = 0
        count += im_abs[ im_abs <= 0.15 * max_abs ].size(0)
    layers_count[ layer_num ] = count
    layers_map_count[ layer_num ] = map_count
    tot_count += count
    tot_map_count += map_count
    if( layer_num % num_of_layers == 15):
        layer_num = 0
    else:
        layer_num += 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = '../../../../data', train = False, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 0)

# resnet18 = ResNet18() # Insert PATH to Model
PATH = './resnet18_epoch_14.pth'
resnet18 = torch.load(PATH)
resnet18.eval()

layers = list(resnet18._modules.items())
# print(layers)

layers[2][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[2][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[2][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[2][1][1].conv2.register_forward_pre_hook(forward_hook_pre)
layers[3][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[3][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[3][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[3][1][1].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[4][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[4][1][1].conv2.register_forward_pre_hook(forward_hook_pre)
layers[5][1][0].conv1.register_forward_pre_hook(forward_hook_pre)
layers[5][1][0].conv2.register_forward_pre_hook(forward_hook_pre)
layers[5][1][1].conv1.register_forward_pre_hook(forward_hook_pre)
layers[5][1][1].conv2.register_forward_pre_hook(forward_hook_pre)

resnet18 = resnet18.to(device)

correct = 0.0
total = 0.0
tot_map_drop = 0

print("Testing Has Started!!!!!!!")

with torch.no_grad():
    for i, data in enumerate(testloader, 1):
        tot_count = 0
        tot_map_count = 0
        images, labels = data[0].to(device), data[1].to(device)
        outputs = resnet18.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        tot_map_drop += tot_count/tot_map_count
        for k in range(num_of_layers + 1):
            if( k == 0 ):
                tot_flop_drop_layer[ k ] += 1 - ( ( layers_map_count[ k ] - layers_count[ k ] ) / layers_map_count[ k ] )
            elif( k == num_of_layers ):
                tot_flop_drop_layer[ k ] += 1 - ( ( layers_map_count[ k - 1 ] - layers_count[ k - 1 ] ) / layers_map_count[ k - 1 ] )
            else:
                tot_flop_drop_layer[ k ] += 1 - ( ( ( layers_map_count[ k - 1 ] - layers_count[ k - 1 ] )( layers_map_count[ k ] - layers_count[ k ] ) ) / ( layers_map_count[ k - 1 ] * layers_map_count[ k ] ) )
        correct += (predicted == labels).sum().item()
        print('Count = ', tot_count, 'Map Count = ', tot_map_count, 'Accuracy = ', (predicted == labels).sum().item()/labels.size(0) )
        # print(((predicted == labels).sum().item())/labels.size(0))
        if(i%200 == 0):
            print('Mini-Batch Count: %4d' % i)
    print('Finished Testing!!!!!!')
    print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)
    print('Total FLOP Drop Percentage as an Average:= ', [ ( j / i ) * 100 for j in tot_flop_drop_layer ])

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
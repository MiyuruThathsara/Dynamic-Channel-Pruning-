import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# Variable Definitions
num_labels = 1000
label_count = [0] * num_labels
num_of_train_imgs = 10000
num_of_test_imgs = 50000
num_of_conv_layers = 17
num_of_int_layers = 3
# num_of_layers = 4
# num_of_blocks_per_layer = 2
# num_of_conv_layers_per_block = 2
data_root = './gdrive/My Drive/Research Project/Data'
batch_size = 64
running_mean_list = []
count_all_list = []
map_count_all_list = []
threshold_mask = []
layer_num = 0
threshold = 0.01598449882655879
mean_scale = 1
mask_flag = None
data_size = batch_size

# Relative PATH to dataset folder
data_root_train = '../../../../data/ImageNet_Cal_set_train'
data_root_test = '../../../../data/ImageNet_Cal_set_test'

# Dataset Preprocessing
data_transform = transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
ImageNet_Cal_dataset_train = datasets.ImageFolder(root=data_root_train, transform=data_transform)
train_dataset_loader = torch.utils.data.DataLoader(ImageNet_Cal_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
ImageNet_Cal_dataset_test = datasets.ImageFolder(root=data_root_test, transform=data_transform)
test_dataset_loader = torch.utils.data.DataLoader(ImageNet_Cal_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

# Initializing the Network Model
# model link: https://download.pytorch.org/models/resnet18-5c106cde.pth
# model link: https://download.pytorch.org/models/resnet34-333f7ec4.pth
PATH = '../../../../data/resnet18-5c106cde.pth'
net = models.resnet18(pretrained=False)
model = torch.load(PATH)
net.load_state_dict(model)
net.eval()
net.to(device)

# Function definitions
def running_mean_var_thresholding_and_masking_with_fixed_mean_single_iter(self, input):
    global running_mean_list
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
        input[0][ var <= max_var.unsqueeze(dim=1) * threshold ] = 0
        count_all_list.append(count_list)
        map_count_all_list.append(map_count_list)

def running_mean_abs_thresholding_and_masking_with_fixed_mean_single_iter(self, input):
    global running_mean_list
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
        input[0][ x_abs <= max_abs.unsqueeze(dim=1) * threshold ] = 0
        count_all_list.append(count_list)
        map_count_all_list.append(map_count_list)

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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Finished Testing!!!!!')
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

def static_mean_accuracy_drop_calc_single_iter(testloader, num_of_conv_layers = num_of_conv_layers, train = True):
    global count_all_list
    global map_count_all_list
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
            outputs = net(images)
            for k in range(num_of_conv_layers):
                for l in range(data_size):
                    if( k == 0 ):
                        tot_flop_drop_layer[ k ] += 0
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) / map_count_all_list[ k - 1 ][ l ] )
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        mask_flag = None
        if(train == True):
            num_of_imgs = num_of_train_imgs
        else:
            num_of_imgs = num_of_test_imgs
    return 100 * correct / total, [ ( j / num_of_imgs ) * 100 for j in tot_flop_drop_layer ]

def static_mean_accuracy_drop_calc(testloader, num_of_conv_layers = num_of_conv_layers, num_of_int_layers = num_of_int_layers, train = True):
    global count_all_list
    global map_count_all_list
    global threshold_mask
    global layer_num
    global mask_flag
    global data_size
    correct = 0.0
    total = 0.0
    tot_flop_drop_layer = [ 0 ] * num_of_conv_layers
    tot_flop_drop_int_layer = [ 0 ] * num_of_int_layers
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
                    elif( k == num_of_conv_layers - 2 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) / map_count_all_list[ k - 1 ][ l ] )
                    elif(k == num_of_conv_layers - 1):
                        tot_flop_drop_layer[ k ] += 0
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) * ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) ) / ( map_count_all_list[ k - 1 ][ l ] * map_count_all_list[ k ][ l ] ) )
                    # ResNet18
                    if(k == 6):
                        tot_flop_drop_int_layer[ 0 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    elif(k == 10):
                        tot_flop_drop_int_layer[ 1 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    elif(k == 14):
                        tot_flop_drop_int_layer[ 2 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    # ResNet34
                    # if(k == 8):
                    #     tot_flop_drop_int_layer[ 0 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    # elif(k == 16):
                    #     tot_flop_drop_int_layer[ 1 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    # elif(k == 28):
                    #     tot_flop_drop_int_layer[ 2 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        mask_flag = None
        if(train == True):
            num_of_imgs = num_of_train_imgs
        else:
            num_of_imgs = num_of_test_imgs
    return 100 * correct / total, [ ( j / num_of_imgs ) * 100 for j in tot_flop_drop_layer ], [ ( l / num_of_imgs ) * 100 for l in tot_flop_drop_int_layer ]

def avg_flop_drop_calc_single_iter(percent, num_of_conv_layers = num_of_conv_layers):
    # ResNet18
    data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)])
    # ResNet34
    # data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_2', (7,7,3,3,512,512,1)])
    int_data = (['layer2.0.shortcut.0', (56,56,1,1,64,128,2)], ['layer3.0.shortcut.0', (28,28,1,1,128,256,2)], ['layer4.0.shortcut.0', (14,14,1,1,256,512,2)])
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

def avg_flop_drop_calc(percent, int_percent, num_of_conv_layers = num_of_conv_layers):
    # ResNet18
    data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)])
    # ResNet34
    # data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_2', (7,7,3,3,512,512,1)])
    int_data = (['layer2.0.shortcut.0', (56,56,1,1,64,128,2)], ['layer3.0.shortcut.0', (28,28,1,1,128,256,2)], ['layer4.0.shortcut.0', (14,14,1,1,256,512,2)])
    tot_flop = 0
    tot_drop_flop = 0
    with torch.no_grad():
        for i in range(num_of_conv_layers):
            flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5] / data[i][1][6]
            drop_flop = percent[i] * flop / 100
            tot_flop += flop
            tot_drop_flop += drop_flop
        for j in range(len(int_percent)):
            flop = 2 * int_data[j][1][0] * int_data[j][1][1] * int_data[j][1][2] * int_data[j][1][3] * int_data[j][1][4] * int_data[j][1][5] / int_data[j][1][6]
            drop_flop = int_percent[j] * flop / 100
            tot_flop += flop
            tot_drop_flop += drop_flop
        flop_drop_percent = tot_drop_flop / tot_flop * 100
    return flop_drop_percent

for name, module in net.named_modules():
    if(isinstance(module, nn.Conv2d)):
        if( name != 'conv1' and name != 'layer2.0.downsample.0' and name != 'layer3.0.downsample.0' and name != 'layer4.0.downsample.0' and name != 'layer4.1.conv2' ):
            module.register_forward_pre_hook(running_mean_var_thresholding_and_masking_with_fixed_mean)

###############################################################################
# Single Iteration
###############################################################################
# print('Calibration Dataset Accuracy Calculation!!!!!!!!!!!!!')
# exp_accuracy = dataset_accuracy_calc(train_dataset_loader)
# print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))

# print('Static Mean Calculation Started!!!!!!!!!!!!')
# mean = static_mean_calc(train_dataset_loader)

# print('Threshold Learning for Calibration Dataset!!!!!!!')
# info = []
# accuracy_drop_list = []
# mean_scale_list = []
# threshold_list = []
# while(True):
#     state = False
#     while(state == False):
#         actual_accuracy, percent = static_mean_accuracy_drop_calc_single_iter(train_dataset_loader)
#         print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
#         target_FLOP_drop_percent = 20
#         const = 0.001
#         flop_drop_percent = avg_flop_drop_calc_single_iter(percent)
#         target_drop_error = flop_drop_percent - target_FLOP_drop_percent
#         print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
#         print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
#         # print('Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent))
#         print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
#         print('Accuracy Drop: {}'.format(exp_accuracy - actual_accuracy))
#         if(abs(target_drop_error) <= 1):
#             state = True
#             print('Optimum Threshold Value: {}'.format(threshold))
#             # print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
#             # print('FLOP drop Error for Cal Dataset: {}'.format(target_drop_error))
#             # print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
#         else:
#             threshold = threshold - const * target_drop_error
#             print( 'Threshold: {}'.format(threshold) )
#             print('###########################################################################')
#             state = False
#     pre_mean_scale = mean_scale
#     mean_scale -= 0.1
#     accuracy_drop = exp_accuracy - actual_accuracy
#     if(mean_scale < -1.0):
#         min_accuracy_index = accuracy_drop_list.index(min(accuracy_drop_list))
#         mean_scale = mean_scale_list[min_accuracy_index]
#         threshold = threshold_list[min_accuracy_index]
#         print('Information List: {}'.format(info[min_accuracy_index]))
#         break
#     else:
#         accuracy_drop_list.append(accuracy_drop)
#         info.append(['Accuracy of the network on the 10000 Calibration images: %.2f %%' % (actual_accuracy), 
#                       'AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent), 
#                       'Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent),
#                     #   'Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent),
#                       'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
#                       'Optimum Mean Scale: {}'.format(pre_mean_scale),
#                       'Optimum Threshold: {}'.format(threshold)])
#         mean_scale_list.append(pre_mean_scale)
#         threshold_list.append(threshold)

# print('Learned Threshold Check for Test Dataset!!!!!!!')
# baseline_test_accuracy = dataset_accuracy_calc(test_dataset_loader)
# test_dataset_accuracy_drop = static_mean_accuracy_drop_calc_single_iter(test_dataset_loader, train = False)
# print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
# print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
# print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc_single_iter(test_dataset_accuracy_drop[1])))

###############################################################################
# Double Iteration
###############################################################################
print('Calibration Dataset Accuracy Calculation!!!!!!!!!!!!!')
exp_accuracy = dataset_accuracy_calc(train_dataset_loader)
print('Expected Accuracy for Calibration Dataset: {}'.format(exp_accuracy))

print('Static Mean Calculation Started!!!!!!!!!!!!')
mean = static_mean_calc(train_dataset_loader)

print('Threshold Learning for Calibration Dataset!!!!!!!')
info = []
accuracy_drop_list = []
mean_scale_list = []
threshold_list = []
while(True):
    state = False
    while(state == False):
        actual_accuracy, percent, int_percent = static_mean_accuracy_drop_calc(train_dataset_loader)
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
                      'Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent),
                      'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
                      'Optimum Mean Scale: {}'.format(pre_mean_scale),
                      'Optimum Threshold: {}'.format(threshold)])
        mean_scale_list.append(pre_mean_scale)
        threshold_list.append(threshold)

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(test_dataset_loader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc(test_dataset_loader, train = False)
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1],test_dataset_accuracy_drop[2])))

# Results
###################################################################################################################
# Single Iteration ABS method
# Accuracy of the network on the 10000 Calibration images: 52.58 %
# AVG FLOP drop percentage for Cal Dataset: 19.237293007425745
# Layer by Layer FLOP Drop for Cal Dataset: [0.0, 23.22546875, 20.8221875, 9.42390625, 15.2065625, 3.9553125000000002, 14.63875, 4.523515625, 5.08984375, 2.5173437499999998, 9.603203125, 13.5471484375, 32.616015625, 16.3049609375, 51.57839843749999, 52.794589843749996, 59.26109375000001]
# FLOP drop Error for Cal Dataset: -0.7627069925742553
# Optimum Mean Scale: -0.5
# Optimum Threshold: 0.12566754833150928
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 17.636000000000003
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 23.2313125, 20.7828125, 9.42615625, 15.201843749999998, 3.95328125, 14.641703125, 4.5470781250000005, 
# 5.150625000000001, 2.557140625, 9.7126796875, 13.6757109375, 32.7475546875, 16.4301875, 51.67741796874999, 52.91783984375, 59.2915]
# Avg FLOP Drop Percentages for Test Dataset: 19.277882673267328

# Double Iteration ABS method without last layer
# Accuracy of the network on the 10000 Calibration images: 68.07 %
# AVG FLOP drop percentage for Cal Dataset: 19.227748453141498
# Layer by Layer FLOP Drop for Cal Dataset: [20.65140625, 29.995847167968748, 16.2763525390625, 12.17727783203125, 8.5292919921875, 3.9237646484375, 4.512257690429688, 4.01375, 2.8523962402343748, 4.286492614746094, 10.48444564819336, 23.798028106689454, 23.484791259765625, 36.60324394226075, 56.21841239929199, 36.6300390625, 0.0]
# Intermediate Layer wise FLOP Drop for Cal Dataset: [1.736328125, 7.1367578125, 36.6300390625]
# FLOP drop Error for Cal Dataset: -0.7722515468585023
# Optimum Mean Scale: -0.09999999999999987
# Optimum Threshold: 0.06826014513770337
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 1.8220000000000027
# Layer by Layer FLOP Drop Percentages for Test dataset: [20.6778125, 30.0139345703125, 16.280447265625, 12.1531123046875, 8.486499023437501, 3.93812744140625, 4.546138671875, 4.0539697265625, 2.8861405029296874, 4.3868690185546875, 10.630404907226563, 23.900352294921877, 23.585906311035156, 36.79435720825195, 56.344272071838375, 36.71884765625, 0.0]
# Avg FLOP Drop Percentages for Test Dataset: 19.277446676173714

# Double Iteration SD method without last layer
# Accuracy of the network on the 10000 Calibration images: 68.15 %
# AVG FLOP drop percentage for Cal Dataset: 19.721375563276364
# Layer by Layer FLOP Drop for Cal Dataset: [23.0171875, 34.13633056640625, 21.025075683593748, 13.47314697265625, 8.10849853515625, 6.871971435546875, 6.088639526367187, 2.620048828125, 1.694898681640625, 1.9545132446289062, 6.058140106201172, 19.556983947753906, 19.914715728759766, 35.30051124572754, 55.40557361602784, 35.3853125, 0.0]
# Intermediate Layer wise FLOP Drop for Cal Dataset: [1.11359375, 4.4297265625, 35.3853125]
# FLOP drop Error for Cal Dataset: -0.2786244367236357
# Optimum Mean Scale: -0.8999999999999998
# Optimum Threshold: 0.0229963138723078
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 2.0600000000000023
# Layer by Layer FLOP Drop Percentages for Test dataset: [23.02728125, 34.14909765625, 21.0391015625, 13.458987792968749, 8.098017578124999, 6.925877197265624, 6.129564331054688, 2.6334942626953124, 1.70849365234375, 1.9947378540039065, 6.132057800292968, 19.647746032714846, 20.01603564453125, 35.42003825378418, 55.53432619476318, 35.517394531250005, 0.0]
# Avg FLOP Drop Percentages for Test Dataset: 19.763009475995535
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
data_size = 224*224*3
label_count = [0] * num_labels
num_of_train_imgs = 10000
num_of_test_imgs = 50000
num_of_conv_layers = 33
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
threshold = 0.056791663889694345
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
PATH = '../../../../data/resnet34-333f7ec4.pth'
net = models.resnet34(pretrained=False)
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

def static_mean_calc(dataloader, num_of_conv_layers= num_of_conv_layers, num_of_cal_imgs= num_of_train_imgs):
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
                running_mean_avg = [ running_mean_avg[j] + running_mean_list[j] for j in range(len(running_mean_avg)) ]
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
                    elif ( k == num_of_conv_layers - 1 ):
                        tot_flop_drop_layer[ k ] += 0
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) * ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) ) / ( map_count_all_list[ k - 1 ][ l ] * map_count_all_list[ k ][ l ] ) )
                    # ResNet18
                    # if(k == 6):
                    #     tot_flop_drop_int_layer[ 0 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    # elif(k == 10):
                    #     tot_flop_drop_int_layer[ 1 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    # elif(k == 14):
                    #     tot_flop_drop_int_layer[ 2 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    # ResNet34
                    if(k == 8):
                        tot_flop_drop_int_layer[ 0 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    elif(k == 16):
                        tot_flop_drop_int_layer[ 1 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    elif(k == 28):
                        tot_flop_drop_int_layer[ 2 ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
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
    # data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)])
    # ResNet34
    data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_2', (7,7,3,3,512,512,1)])
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
    # data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)])
    # ResNet34
    data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_1_conv_2', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_1', (56,56,3,3,64,64,1)], ['layer1_block_2_conv_2', (56,56,3,3,64,64,1)], ['layer2_block_0_conv_1', (56,56,3,3,64,128,2)], ['layer2_block_0_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_1_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_2_conv_2', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_1', (28,28,3,3,128,128,1)], ['layer2_block_3_conv_2', (28,28,3,3,128,128,1)], ['layer3_block_0_conv_1', (28,28,3,3,128,256,2)], ['layer3_block_0_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_1_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_2_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_3_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_4_conv_2', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_1', (14,14,3,3,256,256,1)], ['layer3_block_5_conv_2', (14,14,3,3,256,256,1)], ['layer4_block_0_conv_1', (14,14,3,3,256,512,2)], ['layer4_block_0_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_1_conv_2', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_1', (7,7,3,3,512,512,1)], ['layer4_block_2_conv_2', (7,7,3,3,512,512,1)])
    int_data = (['layer2.0.shortcut.0', (56,56,1,1,64,128,2)], ['layer3.0.shortcut.0', (28,28,1,1,128,256,2)], ['layer4.0.shortcut.0', (14,14,1,1,256,512,2)])
    tot_flop = 0
    tot_drop_flop = 0
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
        if( name != 'conv1' and name != 'layer2.0.downsample.0' and name != 'layer3.0.downsample.0' and name != 'layer4.0.downsample.0' ):
            module.register_forward_pre_hook(running_mean_var_thresholding_and_masking_with_fixed_mean_single_iter)

#################################################################################
# Single Iteration
#################################################################################
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
        actual_accuracy, percent = static_mean_accuracy_drop_calc_single_iter(train_dataset_loader)
        print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
        target_FLOP_drop_percent = 20
        const = 0.001
        flop_drop_percent = avg_flop_drop_calc_single_iter(percent)
        target_drop_error = flop_drop_percent - target_FLOP_drop_percent
        print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
        print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
        # print('Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent))
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
                    #   'Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent),
                      'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
                      'Optimum Mean Scale: {}'.format(pre_mean_scale),
                      'Optimum Threshold: {}'.format(threshold)])
        mean_scale_list.append(pre_mean_scale)
        threshold_list.append(threshold)

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(test_dataset_loader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc_single_iter(test_dataset_loader, train = False)
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc_single_iter(test_dataset_accuracy_drop[1])))

#################################################################################
# Double Iteration
#################################################################################
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
#         actual_accuracy, percent, int_percent = static_mean_accuracy_drop_calc(train_dataset_loader)
#         print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
#         target_FLOP_drop_percent = 20
#         const = 0.001
#         flop_drop_percent = avg_flop_drop_calc(percent, int_percent)
#         target_drop_error = flop_drop_percent - target_FLOP_drop_percent
#         print('AVG FLOP drop percentage for Cal Dataset: {}'.format(flop_drop_percent))
#         print('Layer by Layer FLOP Drop for Cal Dataset: {}'.format(percent))
#         print('Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent))
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
#                       'Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent),
#                       'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
#                       'Optimum Mean Scale: {}'.format(pre_mean_scale),
#                       'Optimum Threshold: {}'.format(threshold)])
#         mean_scale_list.append(pre_mean_scale)
#         threshold_list.append(threshold)

# print('Learned Threshold Check for Test Dataset!!!!!!!')
# baseline_test_accuracy = dataset_accuracy_calc(test_dataset_loader)
# test_dataset_accuracy_drop = static_mean_accuracy_drop_calc(test_dataset_loader, train = False)
# print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
# print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
# print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1],test_dataset_accuracy_drop[2])))

# Resluts
#############################################################################################
# Double Iteration ABS approach
# Accuracy of the network on the 10000 Calibration images: 72.87 %
# AVG FLOP drop percentage for Cal Dataset: 19.22152250076572
# Layer by Layer FLOP Drop for Cal Dataset: [23.41875, 32.148681640625, 19.3104296875, 19.80025146484375, 14.124985351562499, 6.02889892578125, 3.9594873046875, 1.3066833496093748, 3.090989990234375, 5.739467163085937, 4.148699340820312, 0.9590386962890625, 0.7792443847656251, 0.530167236328125, 0.47729064941406246, 0.893192138671875, 2.3686219787597653, 12.599815521240235, 12.180243225097657, 14.843760833740236, 14.615868988037109, 29.4279020690918, 29.660309448242188, 38.69355377197265, 38.96424957275391, 12.409611816406251, 16.071821746826174, 17.530968933105466, 21.497079467773435, 49.62441368103027, 51.532033348083495, 52.081226158142094, 45.11806640625]
# Intermediate Layer wise FLOP Drop for Cal Dataset: [2.02984375, 1.62015625, 11.00220703125]
# FLOP drop Error for Cal Dataset: -0.7784774992342811
# Optimum Mean Scale: -0.2999999999999999
# Optimum Threshold: 0.0731460584474813
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 0.8220000000000027
# Layer by Layer FLOP Drop Percentages for Test dataset: [23.431968750000003, 32.14761767578125, 19.27532861328125, 19.75715966796875, 14.103076171875001, 6.02063720703125, 3.9474892578125, 1.320214111328125, 3.1285510253906255, 5.777634399414063, 4.171948974609375, 0.9659239501953125, 0.7798763427734375, 0.52949853515625, 0.477542724609375, 0.9141370849609375, 2.4245985107421877, 12.73442315673828, 12.313149475097656, 14.875254089355469, 14.631348358154298, 29.465218872070313, 29.699198974609374, 38.803729827880865, 39.07089529418945, 12.426473114013673, 16.101868713378906, 17.61192121887207, 21.528247863769533, 49.55833853149414, 51.42791542816162, 52.05123310852051, 45.128546875]
# Avg FLOP Drop Percentages for Test Dataset: 19.239822821655046

# Double Iteration SD approach
# Accuracy of the network on the 10000 Calibration images: 72.77 %
# AVG FLOP drop percentage for Cal Dataset: 19.54619002519066
# Layer by Layer FLOP Drop for Cal Dataset: [24.44109375, 32.41880859375, 18.803486328125, 17.143447265625, 11.6668798828125, 5.32524658203125, 2.9622509765625, 1.4267248535156252, 2.7187890625, 5.836792602539063, 4.296207885742188, 1.357655029296875, 1.1572320556640625, 0.662716064453125, 0.5822747802734375, 1.1675527954101563, 3.427989501953125, 13.973962860107422, 13.427499237060548, 17.06749099731445, 16.81593292236328, 23.758399963378906, 23.85747634887695, 25.700988769531254, 25.979187774658204, 14.201384887695312, 16.57390441894531, 20.988620223999025, 32.026365432739254, 55.39597381591796, 58.647962493896486, 61.464196395874026, 49.96125]
# Intermediate Layer wise FLOP Drop for Cal Dataset: [2.07265625, 2.479140625, 18.7330859375]
# FLOP drop Error for Cal Dataset: -0.4538099748093387
# Optimum Mean Scale: -0.3999999999999999
# Optimum Threshold: 0.015253099071968769
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 0.9440000000000026
# Layer by Layer FLOP Drop Percentages for Test dataset: [24.46115625, 32.437637695312496, 18.81106787109375, 17.12830615234375, 11.654982421875, 5.32789208984375, 2.9653935546875, 1.455300537109375, 2.7575472412109376, 5.855653320312499, 4.300060791015625, 1.3625787353515626, 1.1570975341796874, 0.6701573486328125, 0.5889075927734375, 1.1884625244140625, 3.4778632812500003, 14.09909393310547, 13.551345764160155, 17.075369384765626, 16.812564544677734, 23.75739001464844, 23.85247033691406, 25.738884948730465, 26.01745556640625, 14.22798385620117, 16.617398223876954, 21.07868014526367, 32.03075645446777, 55.355760070800784, 58.581163520812986, 61.44774236297608, 49.99619140625]
# Avg FLOP Drop Percentages for Test Dataset: 19.564482297536568

# Double Iteraion ABS approach without last layer
# Accuracy of the network on the 10000 Calibration images: 72.32 %
# AVG FLOP drop percentage for Cal Dataset: 19.095522034644013
# Layer by Layer FLOP Drop for Cal Dataset: [23.52921875, 32.91525390625, 20.1240185546875, 20.54068603515625, 15.016098632812499, 6.8989794921875, 4.8765087890625, 1.74602294921875, 3.863606567382812, 7.874976806640626, 6.071294555664062, 1.6385711669921876, 1.3805767822265624, 0.83444580078125, 0.706424560546875, 1.7245574951171874, 3.9895970153808595, 16.219854736328124, 15.573215942382813, 18.64558853149414, 18.3155224609375, 32.38944366455078, 32.6353157043457, 39.97414413452148, 40.38974365234375, 16.495327606201172, 20.61683609008789, 22.29590774536133, 28.751194725036623, 55.35297863006592, 57.270149040222165, 19.45658203125, 0.0]
# Intermediate Layer wise FLOP Drop for Cal Dataset: [2.5462499999999997, 2.554375, 15.843535156249999]
# FLOP drop Error for Cal Dataset: -0.9044779653559871
# Optimum Mean Scale: -0.19999999999999987
# Optimum Threshold: 0.07192021452440098
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 1.3400000000000034
# Layer by Layer FLOP Drop Percentages for Test dataset: [23.550562499999998, 32.933196777343746, 20.11351953125, 20.50859765625, 14.996462890625, 6.89103515625, 4.864574707031251, 1.7651154785156251, 3.91867822265625, 7.927651855468749, 6.09791357421875, 1.6486181640625, 1.3850531005859374, 0.8361944580078126, 0.708227294921875, 1.7697673339843751, 4.081267700195312, 16.36482028198242, 15.710482818603516, 18.674237976074217, 18.333134796142577, 32.42115228271484, 32.66347113037109, 40.06758981323242, 40.488156188964844, 16.538141174316408, 20.666989868164062, 22.391690017700196, 28.761591979980466, 55.28296469116211, 57.17769535064697, 19.38992578125, 0.0]  
# Avg FLOP Drop Percentages for Test Dataset: 19.119612183184216

# Double Iteration SD approach without last layer
# Accuracy of the network on the 10000 Calibration images: 72.45 %
# AVG FLOP drop percentage for Cal Dataset: 19.008874564565982
# Layer by Layer FLOP Drop for Cal Dataset: [25.186406249999997, 34.1619091796875, 21.5414599609375, 20.67449951171875, 14.561474609374999, 6.5506103515625, 3.98663818359375, 2.289312744140625, 3.745623779296875, 7.039645996093751, 4.9747552490234375, 1.6457659912109373, 1.3699389648437499, 0.8062493896484375, 0.6918243408203125, 1.3324295043945313, 3.8812693786621097, 15.508701629638672, 14.866708221435546, 19.13956787109375, 18.9072509765625, 26.732588653564456, 26.923424377441407, 30.389883728027346, 30.726102600097654, 16.241146392822266, 19.158175659179687, 23.552984085083008, 34.70705181121826, 58.77847915649414, 61.96886157989502, 26.803691406250003, 0.0]
# Intermediate Layer wise FLOP Drop for Cal Dataset: [2.79296875, 2.821640625, 20.40158203125]
# FLOP drop Error for Cal Dataset: -0.991125435434018
# Optimum Mean Scale: -0.4999999999999999
# Optimum Threshold: 0.01835067717542163
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 1.318000000000012
# Layer by Layer FLOP Drop Percentages for Test dataset: [25.21246875, 34.201688964843754, 21.5623740234375, 20.646750488281253, 14.53445361328125, 6.552839355468749, 3.984919921875, 2.305506591796875, 3.799386962890625, 7.063773315429687, 4.974307006835938, 1.6519374999999998, 1.3661376953125, 0.8156181640625, 0.700237060546875, 1.3562420043945314, 3.9404374084472655, 15.639615325927734, 14.988667327880858, 19.141967834472656, 18.899203521728516, 26.7368232421875, 26.927417755126953, 30.421574829101562, 30.756196533203124, 16.26877737426758, 19.19634002685547, 23.632666275024413, 34.698651321411134, 58.750309303283686, 61.91229808044434, 26.677933593749998, 0.0]
# Avg FLOP Drop Percentages for Test Dataset: 19.02553121405027

# Single Iteration ABS approach
# Accuracy of the network on the 10000 Calibration images: 62.05 %
# AVG FLOP drop percentage for Cal Dataset: 19.277319495219352
# Layer by Layer FLOP Drop for Cal Dataset: [0.0, 25.377187499999998, 17.6590625, 12.852812499999999, 24.60609375, 5.457343750000001, 11.632031249999999, 4.26484375, 5.270703125, 6.971328124999999, 11.90265625, 4.280937499999999, 3.4690625, 2.715390625, 1.982734375, 1.955859375, 7.2325390625, 8.329531249999999, 25.630976562500003, 6.826484375, 29.619296875, 6.14890625, 44.3880078125, 7.364960937499999, 52.9569921875, 8.974765625, 30.847578125000002, 17.61890625, 35.48291015625, 36.611484374999996, 65.91236328125, 44.07408203125, 69.5146484375]
# FLOP drop Error for Cal Dataset: -0.7226805047806479
# Optimum Mean Scale: -0.19999999999999987
# Optimum Threshold: 0.09836994531545365
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 11.760000000000005
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 25.40015625, 17.648125, 12.870062500000001, 24.609468749999998, 5.4784375, 11.61253125, 4.2696875, 5.28221875, 7.031765625, 11.943859375, 4.32390625, 3.51053125, 2.7495937500000003, 2.0068437500000003, 1.9815937499999998, 7.3355078125, 8.473703125, 25.801265625000003, 6.954429687499999, 29.76184375, 6.232109375, 44.4628671875, 7.439710937500001, 53.108171875, 9.08734375, 30.891234375, 17.757976562499998, 35.699453125000005, 
# 36.57451953125, 65.91772265624999, 43.98614453125, 69.5462890625]
# Avg FLOP Drop Percentages for Test Dataset: 19.32822163245219

# Single Iteration SD approach
# Accuracy of the network on the 10000 Calibration images: 61.13 %
# AVG FLOP drop percentage for Cal Dataset: 19.076074416479187
# Layer by Layer FLOP Drop for Cal Dataset: [0.0, 26.737499999999997, 18.01078125, 15.99046875, 23.45421875, 10.072656250000001, 7.60953125, 8.851875, 4.156953125, 8.253437499999999, 10.803671875, 5.15796875, 3.765703125, 3.5553125, 1.658984375, 2.55171875, 6.363125, 9.146210937500001, 23.6744921875, 8.249921875, 27.329804687499998, 7.7916796875, 35.7153515625, 8.817421875, 39.8184765625, 9.8140234375, 27.326093750000002, 14.982382812500001, 34.7282421875, 39.572265625, 63.63203125, 49.87658203125, 68.928203125]
# FLOP drop Error for Cal Dataset: -0.9239255835208127
# Optimum Mean Scale: -0.3999999999999999
# Optimum Threshold: 0.027151278625352396    
# Learned Threshold Check for Test Dataset!!!!!!!
# Finished Testing!!!!!
# Test Accuracy Drop: 12.918000000000006
# Layer by Layer FLOP Drop Percentages for Test dataset: [0.0, 26.7508125, 18.0165625, 15.99759375, 23.41634375, 10.08090625, 7.59553125, 8.83290625, 4.210687500000001, 8.321984375, 10.82034375, 5.201781250000001, 3.7997343750000003, 3.58296875, 1.711984375, 2.573859375, 6.43646875, 9.232851562499999, 23.8197109375, 8.34571875, 27.4451484375, 7.86290625, 35.8233203125, 8.8755390625, 39.8921875, 9.909890625, 27.3927421875, 15.0751171875, 34.854109375, 39.604, 63.688226562500006, 49.86663671875, 69.02276171875]
# Avg FLOP Drop Percentages for Test Dataset: 19.12208162260967
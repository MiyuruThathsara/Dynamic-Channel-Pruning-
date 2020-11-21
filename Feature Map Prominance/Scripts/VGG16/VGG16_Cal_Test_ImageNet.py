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

# Variable definitions
num_labels = 1000
data_size = 224*224*3
label_count = [0] * num_labels
num_of_cal_imgs = 10000
num_of_conv_layers = 13
batch_size = 4
running_mean_list = []
count_all_list = []
map_count_all_list = []
threshold_mask = []
layer_num = 0
threshold = 0.5257998382593625
mean_scale = 1
mean = []
mask_flag = None

# Relative PATH to dataset folder
data_root_train = '../../../../data/ImageNet_Cal_set_train'
data_root_test = '../../../../data/ImageNet_Cal_set_test'

# Dataset Preprocessing
data_transform = transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
ImageNet_Cal_dataset_train = datasets.ImageFolder(root=data_root_train, transform=data_transform)
train_dataset_loader = torch.utils.data.DataLoader(ImageNet_Cal_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
ImageNet_Cal_dataset_test = datasets.ImageFolder(root=data_root_test, transform=data_transform)
test_dataset_loader = torch.utils.data.DataLoader(ImageNet_Cal_dataset_test, batch_size=batch_size,3 shuffle=False, num_workers=0)

# Initializing the Network Model
# state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth', model_dir=data_root)
PATH = '../../../../data/vgg16-397923af.pth'
net = models.vgg16(pretrained=True)
model = torch.load(PATH)
net.load_state_dict(model)
net.eval()
net.to(device)

# Function definitions
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

def static_mean_calc(dataloader, num_of_conv_layers= num_of_conv_layers, num_of_cal_imgs= num_of_cal_imgs):
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
                running_mean_avg = [ running_mean_avg[j] + running_mean_list[j] for j in range(num_of_conv_layers - 1) ]
            running_mean_list = []
        mask_flag = 1
        running_mean_avg = [ running_mean_avg[k] / num_of_cal_imgs for k in range(num_of_conv_layers - 1) ]
    return running_mean_avg

def static_mean_accuracy_drop_calc(testloader, num_of_conv_layers = num_of_conv_layers):
    global count_all_list
    global map_count_all_list
    global threshold_mask
    global layer_num
    global mask_flag
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
                for l in range(batch_size):
                    if( k == 0 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) / map_count_all_list[ k ][ l ] )
                    elif( k == num_of_conv_layers - 1 ):
                        tot_flop_drop_layer[ k ] += 1 - ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) / map_count_all_list[ k - 1 ][ l ] )
                    else:
                        tot_flop_drop_layer[ k ] += 1 - ( ( ( map_count_all_list[ k - 1 ][ l ] - count_all_list[ k - 1 ][ l ] ) * ( map_count_all_list[ k ][ l ] - count_all_list[ k ][ l ] ) ) / ( map_count_all_list[ k - 1 ][ l ] * map_count_all_list[ k ][ l ] ) )
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        mask_flag = None
    return 100 * correct / total, [ ( j / num_of_cal_imgs ) * 100 for j in tot_flop_drop_layer ]

def avg_flop_drop_calc(percent, num_of_conv_layers = num_of_conv_layers):
    data = (['layer1', (224,224,3,3,3,64)], ['layer2', (224,224,3,3,64,64)], ['layer3', (112,112,3,3,64,128)], ['layer4', (112,112,3,3,128,128)], ['layer5', (56,56,3,3,128,256)], ['layer6', (56,56,3,3,256,256)], ['layer7', (56,56,3,3,256,256)], ['layer8', (28,28,3,3,256,512)], ['layer9', (28,28,3,3,512,512)], ['layer10', (28,28,3,3,512,512)], ['layer11', (14,14,3,3,512,512)], ['layer12', (14,14,3,3,512,512)], ['layer13', (14,14,3,3,512,512)])
    tot_flop = 0
    tot_drop_flop = 0
    for i in range(num_of_conv_layers):
        flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5]
        drop_flop = percent[i] * flop / 100
        tot_flop += flop
        tot_drop_flop += drop_flop
    flop_drop_percent = tot_drop_flop / tot_flop * 100
    return flop_drop_percent

# Hooking function
for name, module in net.named_modules():
    if(isinstance(module,nn.Conv2d)):
        if(name != 'features.0'):
            module.register_forward_pre_hook(running_mean_var_thresholding_and_masking_with_fixed_mean)

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
        actual_accuracy, percent = static_mean_accuracy_drop_calc(train_dataset_loader)
        print('Accuracy of the network on the 10000 Calibration images: {}'.format(actual_accuracy))
        target_FLOP_drop_percent = 20
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
                      'Intermediate Layer wise FLOP Drop for Cal Dataset: {}'.format(int_percent),
                      'FLOP drop Error for Cal Dataset: {}'.format(target_drop_error),
                      'Optimum Mean Scale: {}'.format(pre_mean_scale),
                      'Optimum Threshold: {}'.format(threshold)])
        mean_scale_list.append(pre_mean_scale)
        threshold_list.append(threshold)

print('Learned Threshold Check for Test Dataset!!!!!!!')
baseline_test_accuracy = dataset_accuracy_calc(test_dataset_loader)
test_dataset_accuracy_drop = static_mean_accuracy_drop_calc(test_dataset_loader)
print('Test Accuracy Drop: {}'.format(baseline_test_accuracy - test_dataset_accuracy_drop[0]))
print('Layer by Layer FLOP Drop Percentages for Test dataset: {}'.format(test_dataset_accuracy_drop[1]))
print('Avg FLOP Drop Percentages for Test Dataset: {}'.format(avg_flop_drop_calc(test_dataset_accuracy_drop[1])))
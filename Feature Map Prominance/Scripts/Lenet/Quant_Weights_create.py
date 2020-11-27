import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from Lenet5 import LeNet5
import sys
import pickle

quant_func_file_path = "D:/University Academics/Research_Works/Scripts/Dynamic Pruning Scripts/Feature Map Prominance/Scripts/Quantization_Deployed"
sys.path.append(quant_func_file_path)

from Quant_funcs import *

device = torch.device('cpu')
print('Device: ', device)
PATH = "./lenet5_net_15.pth"
QUANT_BIT_WIDTH = 8
BATCH_SIZE = 32

with open('quantization_weight_values.txt', 'rb') as fm:
    quant_weigh_values = pickle.load(fm)

net = torch.load(PATH)
net.eval()
net.to(device)

bias_quant_facts = [11022.93505603075, 7009.924344209723, 2505.8722660893027, 562.9974571392655, 526.4875848313935]

# Weights and biases quantization to levels
with torch.no_grad():
    params = net.state_dict()
    layer_names = []
    bias_quant_iter = 0
    for weight_name in net.state_dict():
        layer_names.append(weight_name)
        if( weight_name != "feature_extractor.0.bias" and weight_name != "feature_extractor.3.bias" and weight_name != "feature_extractor.6.bias" and weight_name != "classifier.0.bias" and weight_name != "classifier.2.bias" ):
            params[ weight_name ] = quantize_array_to_levels( params[ weight_name ], quant_weigh_values[ weight_name ] )
        else:
            params[ weight_name ] = ( params[ weight_name ] * bias_quant_facts[ bias_quant_iter ] ).type(torch.int)
            bias_quant_iter += 1

    index = 0
    for parameters in net.parameters():
        parameters.data = params[ layer_names[index] ].type(torch.int)
        index += 1

with torch.no_grad():
    iter_ = 0
    layer_num = 1
    for params in net.parameters():
        if( iter_%2 == 0 ):
            # weights
            if( layer_num < 4 ):
                # convolutional layer (shape = (120,16,5,5))
                shape = (120,16,5,5)
                param_shape = params.shape
                value_str = "const q_data weights[120][16][5][5] = {"
                for i1 in range(shape[0]):
                    value_str += "{"
                    for i2 in range(shape[1]):
                        value_str += "{"
                        for i3 in range(shape[2]):
                            value_str += "{"
                            for i4 in range(shape[3]):
                                if( i1 < param_shape[0] and i2 < param_shape[1] ):
                                    value_str += " " + str(params[ i1 ][ i2 ][ i3 ][ i4 ].item()) + ","
                                else:
                                    value_str += " 0,"
                            value_str = value_str[:-1] + "},\n"
                        value_str = value_str[:-2] + "},\n"
                    value_str = value_str[:-2] + "},\n"
                value_str = value_str[:-2] + "};"
                file_name = "conv" + str(layer_num) + "_quant_layer_weights.txt"
                file = open(file_name, 'w')
                file.write(value_str)
                file.close()

            else:
                # fully-connected layer 
                if( layer_num == 4 ):
                    # fc_type1 (shape = (84,120))
                    shape = (84,120)
                    param_shape = params.shape
                    value_str = "const q_data weights[84][120] = {"
                    for i1 in range(shape[0]):
                        value_str += "{"
                        for i2 in range(shape[1]):
                            value_str += " " + str(params[i1][i2].item()) + ","
                        value_str = value_str[:-1] + "},\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type1_quant_weights.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
                else:
                    # fc_type2 (shape = (10,84))
                    shape = (10,84)
                    param_shape = params.shape
                    value_str = "const q_data weights[10][84] = {"
                    for i1 in range(shape[0]):
                        value_str += "{"
                        for i2 in range(shape[1]):
                            value_str += " " + str(params[i1][i2].item()) + ","
                        value_str = value_str[:-1] + "},\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type2_quant_weights.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
        else:
            # biases
            if( layer_num < 4 ):
                # convolutional layer (shape = (120))
                shape = (120,)
                param_shape = params.shape
                value_str = "const acc_data biases[120] = {"
                for i in range(shape[0]):
                    if( i < param_shape[0] ):
                        value_str += " " + str(params[i].item()) + ",\n"
                    else:
                        value_str += "0,\n"
                value_str = value_str[:-2] + "};"
                file_name = "conv" + str(layer_num) + "_quant_layer_biases.txt"
                file = open(file_name, 'w')
                file.write(value_str)
                file.close()
            else:
                # fully-connected layer
                if( layer_num == 4 ):
                    # fc_type1 (shape = (84))
                    shape = (84,)
                    param_shape = params.shape
                    value_str = "const acc_data biases[84] = {"
                    for i in range(shape[0]):
                        value_str += " " + str(params[i].item()) + ",\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type1_quant_biases.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
                else:
                    # fc_type2 (shape = (10))
                    shape = (10,)
                    param_shape = params.shape
                    value_str = "const acc_data biases[10] = {"
                    for i in range(shape[0]):
                        value_str += " " + str(params[i].item()) + ",\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type2_quant_biases.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
            layer_num += 1
        iter_ = iter_ + 1   

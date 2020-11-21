import numpy as np
from datetime import datetime 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

PATH = "./lenet5_net_15.pth"

net = torch.load(PATH)

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
                value_str = "weights[120][16][5][5] = {"
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
                file_name = "conv" + str(layer_num) + "_layer_weights.txt"
                file = open(file_name, 'w')
                file.write(value_str)
                file.close()

            else:
                # fully-connected layer 
                if( layer_num == 4 ):
                    # fc_type1 (shape = (84,120))
                    shape = (84,120)
                    param_shape = params.shape
                    value_str = "weights[84][120] = {"
                    for i1 in range(shape[0]):
                        value_str += "{"
                        for i2 in range(shape[1]):
                            value_str += " " + str(params[i1][i2].item()) + ","
                        value_str = value_str[:-1] + "},\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type1_weights.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
                else:
                    # fc_type2 (shape = (10,84))
                    shape = (10,84)
                    param_shape = params.shape
                    value_str = "weights[10][84] = {"
                    for i1 in range(shape[0]):
                        value_str += "{"
                        for i2 in range(shape[1]):
                            value_str += " " + str(params[i1][i2].item()) + ","
                        value_str = value_str[:-1] + "},\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type2_weights.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
        else:
            # biases
            if( layer_num < 4 ):
                # convolutional layer (shape = (120))
                shape = (120,)
                param_shape = params.shape
                value_str = "biases[120] = {"
                for i in range(shape[0]):
                    if( i < param_shape[0] ):
                        value_str += " " + str(params[i].item()) + ",\n"
                    else:
                        value_str += "0,\n"
                value_str = value_str[:-2] + "};"
                file_name = "conv" + str(layer_num) + "_layer_biases.txt"
                file = open(file_name, 'w')
                file.write(value_str)
                file.close()
            else:
                # fully-connected layer
                if( layer_num == 4 ):
                    # fc_type1 (shape = (84))
                    shape = (84,)
                    param_shape = params.shape
                    value_str = "biases[84] = {"
                    for i in range(shape[0]):
                        value_str += " " + str(params[i].item()) + ",\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type1_biases.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
                else:
                    # fc_type2 (shape = (10))
                    shape = (10,)
                    param_shape = params.shape
                    value_str = "biases[10] = {"
                    for i in range(shape[0]):
                        value_str += " " + str(params[i].item()) + ",\n"
                    value_str = value_str[:-2] + "};"
                    file_name = "fc_type2_biases.txt"
                    file = open(file_name, 'w')
                    file.write(value_str)
                    file.close()
            layer_num += 1
        iter_ = iter_ + 1    